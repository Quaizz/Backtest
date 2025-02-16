# Core data processing
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Database
import duckdb

# Progress tracking and utilities
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import matplotlib
import mplcursors
# Scientific computing
import scipy.stats
import calendar


def load_distribution_data(start_date, end_date, db_conn=None):
    """
    Load distribution data for backtest period
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    db_conn : duckdb.DuckDBPyConnection, optional
        Database connection. If None, creates new connection
        
    Returns:
    --------
    DataFrame : Distribution data from CRSP
    """
    close_conn = False
    if db_conn is None:
        db_conn = duckdb.connect('wrds_data.db', read_only=True)
        close_conn = True
        
    try:
        # Query distribution data
        dist_query = f"""
        SELECT *
        FROM stkdistributions
        WHERE disexdt BETWEEN DATE '{start_date}' AND DATE '{end_date}' 
        ORDER BY disexdt
        """
        
        distribution_data = db_conn.execute(
            dist_query.format(start_date=start_date, end_date=end_date)
            ).fetchdf()
        
    finally:
        if close_conn:
            db_conn.close()
            
    return distribution_data

class CommissionModel:
    """Base class for commission models"""
    def __init__(self):
        pass
        
    def calculate_commission(self, shares: int, price: float) -> float:
        raise NotImplementedError("Subclass must implement calculate_commission")




class InteractiveBrokersModel(CommissionModel):
    def __init__(self, commission_per_share=0.005, min_commission=1.0, max_commission_pct=0.01):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_commission_pct = max_commission_pct
        
    def calculate_commission(self, shares: int, price: float) -> float:
        trade_value = abs(shares * price)
        commission = abs(shares) * self.commission_per_share
        commission = max(commission, self.min_commission)
        max_commission = trade_value * self.max_commission_pct
        commission = min(commission, max_commission)
        return commission





class Trading:
    def __init__(self, cash, commission_model, margin_rate=0.06, is_print=False, txt_file=None):
        """
        Initialize trading system with margin and corporate action support
        """
        self.cash = np.array([cash])
        self.positions = {}  # {symbol: [cost_basis, shares, current_price, entry_date]}
        self.margin_rate = margin_rate
        self.commission_model = commission_model
        
        # Margin tracking
        self.margin_loans = {}  # {date: loan_amount}
        self.margin_interest = []
        
        # Corporate action tracking
        self.pending_cash = {}  # {date: [(amount, source_permno)]}
        self.pending_shares = {}  # {date: [(permno, shares)]}
        
        # Price tracking
        self.last_valid_prices = {}
        self.suspended_stocks = set()
        
        # Performance tracking
        self.daily_value_list = [cash]
        self.daily_margin_usage = []
        self.daily_turnover = []
        self.trades_history = []
        
        # Position count
        self.position_count = 0

        self.total_trade_value = 0

        # Add logging parameters
        self.is_print = is_print
        self.txt_file = txt_file
        
        # Add trading info storage for logging
        self.trading_buy_information = {}
        self.trading_sell_information = {}

    def logger(self, txt_str):
        """Log message to file and optionally print"""
        if self.txt_file is not None:
            print(txt_str, file=self.txt_file)
        if self.is_print:
            print(txt_str)

    def print_position_info(self, current_date, price_df):
        """Print current position information"""
        self.logger(f"{'-'*15} {current_date} Positions {'-'*20}")
        if len(self.positions) > 0:
            position_data = []
            for symbol, pos in self.positions.items():
                cost_basis, shares, current_price, entry_date = pos
                position_data.append({
                    'Symbol': symbol,
                    'Cost': cost_basis,
                    'Shares': shares,
                    'Current_Price': current_price,
                    'Change_pct': (current_price/cost_basis - 1) * 100,
                    'Position_Value': shares * current_price,
                    'Position_pct': (shares * current_price) / self.daily_value_list[-1] * 100
                })
            
            position_df = pd.DataFrame(position_data)
            self.logger(position_df.round(2))
            
        total_value = self.get_portfolio_value()
        self.logger(f"Total Value: {total_value:,.2f}, "
                f"Cash: {self.cash[0]:,.2f}, "
                f"Cash %: {(self.cash[0]/total_value*100):.2f}%")

    def print_trading_info(self, current_date):
        """Print trading information from trades_history"""
        # Get today's trades
        todays_trades = [trade for trade in self.trades_history 
                        if trade['date'] == current_date]
        
        # Separate buys and sells
        buys = [trade for trade in todays_trades if trade['action'] == 'BUY']
        sells = [trade for trade in todays_trades if trade['action'] == 'SELL']
        delists = [trade for trade in todays_trades if trade['action'] == 'DELISTED']

        # Print sells
        if sells:
            self.logger(f"{'-'*15} {current_date} Sells {'-'*20}")
            sell_df = pd.DataFrame(sells)[['symbol', 'shares', 'price', 'commission', 'margin_used']]
            sell_df.columns = ['Symbol', 'Shares', 'Price', 'Commission', 'Margin']
            self.logger(sell_df.round(2))

        # Print buys
        if buys:
            self.logger(f"{'-'*15} {current_date} Buys {'-'*20}")
            buy_df = pd.DataFrame(buys)[['symbol', 'shares', 'price', 'commission', 'margin_used']]
            buy_df.columns = ['Symbol', 'Shares', 'Price', 'Commission', 'Margin']
            self.logger(buy_df.round(2))

        # Print delistings
        if delists:
            self.logger(f"{'-'*15} {current_date} Delistings {'-'*20}")
            delist_df = pd.DataFrame(delists)[['symbol', 'shares', 'last_price', 'return', 'final_value']]
            delist_df.columns = ['Symbol', 'Shares', 'Last Price', 'Return', 'Final Value']
            self.logger(delist_df.round(2))





    def handle_delisting(self, date, dsf_data):
        """Handle delisted stocks before market open"""
        for permno in list(self.positions.keys()):
            if permno not in dsf_data.index:
                continue
                
            if dsf_data.loc[permno, 'dlydelflg'] == 'Y':
                # Get delisting return
                dly_ret = dsf_data.loc[permno, 'dlyret']
                if pd.isna(dly_ret):
                    dly_ret = -1.0
                
                # Calculate final value
                shares = self.positions[permno][1]
                last_price = self.positions[permno][2]
                final_value = shares * last_price * (1 + dly_ret)
                
                # Add to cash and remove position
                self.cash[0] += final_value
                del self.positions[permno]
                self.position_count -= 1
                
                # Log delisting
                self.trades_history.append({
                    'date': date,
                    'symbol': permno,
                    'action': 'DELISTED',
                    'shares': shares,
                    'last_price': last_price,
                    'return': dly_ret,
                    'final_value': final_value
                })

    def handle_events(self, date, dsf_data, distribution_data):
        """Handle all corporate actions before market open"""
        # First handle delistings
        self.handle_delisting(date, dsf_data)
        
        # Process any pending payments due today
        self.process_pending_payments(date)
        
        # Handle corporate actions for remaining positions
        for permno in list(self.positions.keys()):
            if permno not in dsf_data.index:
                continue
                
            dist_flag = dsf_data.loc[permno, 'dlydistretflg']
            
            if dist_flag in ['NO', 'NA', 'T1', 'N1']:
                continue
                
            if dist_flag == 'S2':
                split_factor = dsf_data.loc[permno, 'dlyfacprc']
                self.handle_split(permno, split_factor)
                continue
            
            events = distribution_data[
                (distribution_data['permno'] == permno) & 
                (distribution_data['disexdt'] == date)
            ]
            
            for _, event in events.iterrows():
                self.process_distribution_event(permno, event)

    def process_distribution_event(self, permno, event):
        """Process a single distribution event"""
        if event['distype'] in ['SP', 'SD', 'CD', 'ROC']:
            raw_cash = event['disdivamt'] * self.positions[permno][1]
            tax_rate = self.get_tax_rate(event['distaxtype'])
            net_cash = raw_cash * (1 - tax_rate)
            
            pay_date = event['dispaydt']
            if pay_date not in self.pending_cash:
                self.pending_cash[pay_date] = []
            self.pending_cash[pay_date].append((net_cash, permno))
            
        elif event['distype'] == 'FRS':
            if event['disdetailtype'] == 'STKSPL':
                factor = 1 + event['disfacshr']
                self.handle_split(permno, factor)
                
            elif event['disdetailtype'] == 'STKDIV':
                new_shares = self.positions[permno][1] * event['disfacshr']
                pay_date = event['dispaydt']
                
                if pay_date not in self.pending_shares:
                    self.pending_shares[pay_date] = []
                self.pending_shares[pay_date].append((permno, new_shares))

    def handle_split(self, permno, factor):
        """Handle stock split"""
        if permno in self.positions:
            position = self.positions[permno]
            position[1] *= factor
            position[0] /= factor
            position[2] /= factor
            
            if permno in self.last_valid_prices:
                self.last_valid_prices[permno] /= factor

    def process_pending_payments(self, date):
        """Process pending cash and share payments"""
        if date in self.pending_cash:
            for amount, source_permno in self.pending_cash[date]:
                self.cash[0] += amount
            del self.pending_cash[date]
            
        if date in self.pending_shares:
            for permno, new_shares in self.pending_shares[date]:
                if permno in self.positions:
                    self.positions[permno][1] += new_shares
            del self.pending_shares[date]

    def calculate_margin_interest(self, date):
        """Calculate daily margin interest"""
        daily_rate = self.margin_rate / 365
        margin_used = max(0, -self.cash[0])
        interest = margin_used * daily_rate
        self.margin_interest.append(interest)
        self.cash[0] -= interest
        return interest

    def get_buying_power(self):
        """Calculate available buying power including margin"""
        portfolio_value = self.get_portfolio_value()
        return max(0, portfolio_value * 2 - self.get_total_position_value())

    def order_buy(self, date, symbol, shares, price):
        """Place buy order with margin support"""
        if pd.isna(price) or shares <= 0 or symbol in self.suspended_stocks:
            return False
            
        total_cost = shares * price
        commission = self.commission_model.calculate_commission(shares, price)
        total_cost += commission
        self.total_trade_value += total_cost


        if total_cost > self.get_buying_power():
            return False
            
        if total_cost > self.cash[0]:
            margin_amount = total_cost - self.cash[0]
            self.margin_loans[date] = self.margin_loans.get(date, 0) + margin_amount
            
        if symbol in self.positions:
            old_basis = self.positions[symbol][0]
            old_shares = self.positions[symbol][1]
            new_shares = old_shares + shares
            new_basis = ((old_basis * old_shares) + (price * shares)) / new_shares
            self.positions[symbol] = [new_basis, new_shares, price, date]
        else:
            self.positions[symbol] = [price, shares, price, date]
            self.position_count += 1
            
        self.cash[0] -= total_cost
        self.last_valid_prices[symbol] = price
        
        self.trades_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'commission': commission,
            'margin_used': max(0, -self.cash[0])
        })
        
        return True

    def order_sell(self, date, symbol, shares, price):
        """Place sell order with margin paydown"""
        if symbol not in self.positions or pd.isna(price) or symbol in self.suspended_stocks:
            return False
            
        current_shares = self.positions[symbol][1]
        if shares > current_shares:
            shares = current_shares
            
        commission = self.commission_model.calculate_commission(shares, price)
        proceeds = (shares * price) - commission
        self.total_trade_value += shares * price

        if shares == current_shares:
            del self.positions[symbol]
            self.position_count -= 1
        else:
            self.positions[symbol][1] -= shares
            
        self.cash[0] += proceeds
        self.repay_margin_loans(proceeds, date)
        
        self.last_valid_prices[symbol] = price
        
        self.trades_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'commission': commission,
            'margin_used': max(0, -self.cash[0])
        })
        
        return True

    def repay_margin_loans(self, amount, date):
        """Pay down margin loans with available cash"""
        if amount <= 0 or not self.margin_loans:
            return
            
        for loan_date in sorted(self.margin_loans.keys()):
            if amount <= 0:
                break
                
            loan = self.margin_loans[loan_date]
            payment = min(loan, amount)
            self.margin_loans[loan_date] -= payment
            amount -= payment
            
            if self.margin_loans[loan_date] <= 0:
                del self.margin_loans[loan_date]
    '''
    def update_prices(self, date, price_data, timing='open'):
        """Update prices handling missing values"""
        if timing == 'open':
            price_field = 'dlyopen'
            prev_price_field = 'dlyprevprc'
            flag_field = 'dlyprevprcflg'
        else:  # close
            price_field = 'dlyclose'
            prev_price_field = 'dlyprc'
            flag_field = 'dlyprcflg'
            
        self.suspended_stocks.clear()
        
        for symbol in list(self.positions.keys()):
            if symbol not in price_data.index:
                continue
                
            stock_data = price_data.loc[symbol]
            price_flag = stock_data[flag_field]
            
            if price_flag in ['HA', 'SU', 'MP']:
                self.suspended_stocks.add(symbol)
                if symbol in self.last_valid_prices:
                    self.positions[symbol][2] = self.last_valid_prices[symbol]
                continue
                
            price = stock_data[price_field]
            if pd.isna(price):
                price = stock_data[prev_price_field]
            
            if pd.notna(price):
                self.positions[symbol][2] = price
                self.last_valid_prices[symbol] = price
            elif symbol in self.last_valid_prices:
                self.positions[symbol][2] = self.last_valid_prices[symbol]
    '''

    def update_prices(self, date, price_data, timing='open'):
        """
        Update prices handling missing values
        Modifies the original price_data DataFrame to fill in missing prices
        """
        if timing == 'open':
            price_field = 'dlyopen'
            prev_price_field = 'dlyprevprc'
            flag_field = 'dlyprevprcflg'
        else:  # close
            price_field = 'dlyclose'
            prev_price_field = 'dlyprc'
            flag_field = 'dlyprcflg'
        
        self.suspended_stocks.clear()
        
        # Tracking variables
        total_stocks = len(self.positions)
        imputed_stocks = 0
        suspended_stocks = 0
        
        for symbol in list(self.positions.keys()):
            if symbol not in price_data.index:
                suspended_stocks += 1
                self.suspended_stocks.add(symbol)
                continue
            
            stock_data = price_data.loc[symbol]
            price_flag = stock_data[flag_field]
            
            # Check for suspended or halted stocks
            if price_flag in ['HA', 'SU', 'MP']:
                self.suspended_stocks.add(symbol)
                suspended_stocks += 1
                
                # Use last valid price if available
                if symbol in self.last_valid_prices:
                    price_data.loc[symbol, price_field] = self.last_valid_prices[symbol]
                    self.positions[symbol][2] = self.last_valid_prices[symbol]
                continue
            
            # Check and fill missing prices
            if pd.isna(price_data.loc[symbol, price_field]):
                # Try previous price fields
                alt_price = stock_data[prev_price_field]
                
                if pd.notna(alt_price):
                    price_data.loc[symbol, price_field] = alt_price
                    self.positions[symbol][2] = alt_price
                    self.last_valid_prices[symbol] = alt_price
                    imputed_stocks += 1
                elif symbol in self.last_valid_prices:
                    # Use last known valid price
                    price_data.loc[symbol, price_field] = self.last_valid_prices[symbol]
                    self.positions[symbol][2] = self.last_valid_prices[symbol]
                    imputed_stocks += 1
                else:
                    # If no price is available, skip this symbol
                    self.suspended_stocks.add(symbol)
                    suspended_stocks += 1
                    continue
            else:
                # Update position and last valid prices with the current price
                self.positions[symbol][2] = price_data.loc[symbol, price_field]
                self.last_valid_prices[symbol] = price_data.loc[symbol, price_field]
        
        # Log imputation statistics
        #self.logger(f"Price Update ({timing}): Total Stocks: {total_stocks}, "
        #            f"Imputed: {imputed_stocks}, Suspended: {suspended_stocks}")
        
        return price_data


    def get_portfolio_value(self, include_margin_cost=True):
        """Calculate total portfolio value"""
        value = self.cash[0]
        for symbol, pos in self.positions.items():
            value += pos[1] * pos[2]
            
        if include_margin_cost:
            value -= sum(self.margin_loans.values())
            
        return value

    def get_tax_rate(self, tax_type):
        """Get tax rate based on distribution tax type"""
        tax_rates = {
            'QUALIFIED': 0.15,
            'ORDINARY': 0.25,
        }
        return tax_rates.get(tax_type, 0.15)
    
    def get_total_position_value(self):
        """Calculate total value of current positions"""
        total_value = 0
        for symbol, pos in self.positions.items():
            total_value += pos[1] * pos[2]  # shares * current price
        return total_value



class Backtest:
    def __init__(self, cash, commission_model, data_df_dic, trading_dates, 
             distribution_data, factor_list, dir_='results', save_format='html'):
        """
        Initialize backtest system with pre-loaded data
        
        Parameters:
        -----------
        cash : float
            Initial cash
        commission_model : CommissionModel
            Commission calculation model
        data_df_dic : dict
            Dictionary of daily data {date: DataFrame}
        trading_dates : list
            List of trading dates
        distribution_data : DataFrame
            CRSP distribution data
        dir_ : str
            Directory for saving results
        save_format : str
            Format for saving results ('html', 'xlsx', etc.)
        """
        # Create output directory
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        
        # Initialize file paths
        self.save_format = save_format
        self.txt_file = open(os.path.join(dir_, 'logs.txt'), 'w+')
        self.png_save_path = os.path.join(dir_, f'portfolio_return.{save_format}')
        self.return_df_path = os.path.join(dir_, 'return_df.csv')

        # Store data
        self.data_df_dic = data_df_dic
        self.trading_dates = trading_dates
        self.distribution_data = distribution_data
        self.total_trade_value = 0
        self.factor_list = factor_list if isinstance(factor_list, list) else [factor_list]

        # Initialize parameters
        self.set_parameters()
        
        # Initialize trading system
        self.trading = Trading(cash, commission_model, self.margin_rate)
        
        # Performance tracking
        self.turnover_dic = {}

        self.txt_file = None  # Initialize as None
        if dir_ is not None:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            self.txt_file = open(os.path.join(dir_, 'logs.txt'), 'w+')
        
    def set_parameters(self):
        """Set default backtest parameters"""
        # Portfolio construction
        self.stock_num = 50
        self.margin_rate = 0.06
        
        # Trading rules
        self.rebalance_freq = '1d'  # '1d', '1w', '1m'
        self.rebalance_day = 1      # Day of month/week for rebalancing
        
        # Universe filters
        self.min_price = 2.0
        self.market_cap_percentile = 0.05
        
    def logger(self, txt_str):
        """Log message to file"""
        if self.txt_file is not None and not self.txt_file.closed:
            print(txt_str, file=self.txt_file)
        print(txt_str)  # Also print to console
            
    def get_rebalance_dates(self):
        """Get rebalancing dates based on frequency"""
        if self.rebalance_freq == '1d':
            return self.trading_dates
            
        dates_df = pd.DataFrame({
            'date': pd.to_datetime(self.trading_dates)
        })
        
        if self.rebalance_freq == '1w':
            mask = dates_df['date'].dt.weekday == self.rebalance_day
            return dates_df[mask]['date'].dt.strftime('%Y-%m-%d').tolist()
            
        elif self.rebalance_freq == '1m':
            dates_df['ym'] = dates_df['date'].dt.strftime('%Y%m')
            monthly_groups = dates_df.groupby('ym')
            
            rebalance_dates = []
            for _, group in monthly_groups:
                group = group.sort_values('date')
                target_day = min(self.rebalance_day, group['date'].iloc[-1].day)
                valid_dates = group[group['date'].dt.day <= target_day]
                
                if not valid_dates.empty:
                    rebalance_dates.append(valid_dates.iloc[-1]['date'].strftime('%Y-%m-%d'))
                else:
                    rebalance_dates.append(group.iloc[0]['date'].strftime('%Y-%m-%d'))
                    
            return rebalance_dates
            
    def before_market_open(self, date, current_data_df):
        """Process events and generate targets before market open"""
        try:
            # Handle delistings and corporate actions
            self.trading.handle_events(
                date, 
                current_data_df, 
                self.distribution_data[self.distribution_data['disexdt'] == date]
            )
            
            # Update prices with open prices
            current_data_df = self.trading.update_prices(date, current_data_df, timing='open')
            
            # Generate new portfolio targets on rebalance dates
            if date in self.get_rebalance_dates():
                return self.generate_portfolio_targets(current_data_df)
            return {}
        except Exception as e:
            self.logger(f"Error in before_market_open for {date}: {str(e)}")
            return {}
        
    def generate_portfolio_targets(self, data_df):
        """
        Generate target portfolio based on factor values
        
        Parameters:
        -----------
        data_df : DataFrame
            Daily stock data including factors
            
        Returns:
        --------
        dict : {symbol: target_weight}
        """
        try:
            # Validate data
            if data_df is None or len(data_df) == 0:
                self.logger("Error: Empty dataframe passed to generate_portfolio_targets")
                return {}

            # Filter universe
            try:
                universe_df = data_df[
                    (data_df['dlyprc'] > self.min_price) &
                    (data_df['dlycap'] > data_df['dlycap'].quantile(self.market_cap_percentile))
                ].copy()
            except Exception as filter_err:
                self.logger(f"Error filtering universe: {str(filter_err)}")
                return {}
            
            # Validate factor list
            if not self.factor_list or not isinstance(self.factor_list, list):
                self.logger("Warning: No valid factors specified")
                return {}

            # Calculate composite rank for each factor
            ranks = pd.DataFrame(index=universe_df.index)
            valid_factors = []

            for factor in self.factor_list:
                # Detailed checking for factor
                if factor not in universe_df.columns:
                    self.logger(f"Warning: Factor {factor} not found in data")
                    continue
                
                # Check if factor column has numeric data
                if not pd.api.types.is_numeric_dtype(universe_df[factor]):
                    self.logger(f"Warning: Factor {factor} is not numeric")
                    continue
                
                # Additional check for NaN values
                factor_series = universe_df[factor]
                if factor_series.isna().all():
                    self.logger(f"Warning: All values for factor {factor} are NaN")
                    continue
                
                # Calculate percentile rank, dropping NaNs
                rank_series = factor_series.rank(pct=True, na_option='keep')
                ranks[f"{factor}_rank"] = rank_series
                valid_factors.append(factor)
            
            # Skip if no valid factors
            if len(valid_factors) == 0:
                self.logger("No valid factors found for portfolio construction")
                return {}
            
            # Calculate composite score
            try:
                ranks['composite_score'] = ranks.mean(axis=1)
            except Exception as e:
                self.logger(f"Error calculating composite score: {str(e)}")
                return {}
            
            # Remove rows with NaN composite score
            universe_df['composite_score'] = ranks['composite_score']
            universe_df = universe_df.dropna(subset=['composite_score'])
            
            # Check stock availability
            if len(universe_df) < self.stock_num:
                self.logger(f"Warning: Only {len(universe_df)} stocks available, less than target {self.stock_num}")
                if len(universe_df) == 0:
                    return {}
            
            # Select top stocks
            universe_df = universe_df.sort_values('composite_score', ascending=False)
            selected_stocks = universe_df.head(self.stock_num).index
            
            # Equal weight portfolio
            weight = 1.0 / len(selected_stocks)
            target_portfolio = {stock: weight for stock in selected_stocks}
            
            # Log portfolio construction details
            self.logger(f"Portfolio Construction: {len(selected_stocks)} stocks selected")
            self.logger(f"Factors used: {', '.join(valid_factors)}")
            
            return target_portfolio
        except Exception as e:
            self.logger(f"Unexpected error in generate_portfolio_targets: {str(e)}")
            return {}
        
    def calculate_shares_from_weight(self, weight, symbol, price_df):
        """Calculate number of shares from target weight"""
        try:
            # Validate inputs
            if price_df is None or symbol not in price_df.index:
                self.logger(f"Warning: Symbol {symbol} not found in price data")
                return 0
            
            portfolio_value = self.trading.get_portfolio_value()
            
            # Check for NaN values
            if pd.isna(weight):
                self.logger(f"Warning: NaN weight for symbol {symbol}")
                return 0
            
            # Try to get price with fallback mechanisms
            price = price_df.loc[symbol, 'dlyopen']
            
            # Check for invalid price, try alternatives
            if pd.isna(price) or price <= 0:
                # Try alternative price sources
                if not pd.isna(price_df.loc[symbol, 'dlyprevprc']) and price_df.loc[symbol, 'dlyprevprc'] > 0:
                    price = price_df.loc[symbol, 'dlyprevprc']
                elif not pd.isna(price_df.loc[symbol, 'dlyclose']) and price_df.loc[symbol, 'dlyclose'] > 0:
                    price = price_df.loc[symbol, 'dlyclose']
                else:
                    self.logger(f"Warning: Invalid price {price} for symbol {symbol}")
                    return 0
            
            # Calculate target value
            target_value = portfolio_value * weight
            
            # Ensure target_value is a valid number
            if pd.isna(target_value):
                self.logger(f"Warning: Unable to calculate target value for {symbol}")
                return 0
            
            # Calculate shares, rounding to nearest 100
            try:
                shares = int(target_value / price / 100) * 100
                return max(0, shares)  # Ensure non-negative
            except Exception as calc_err:
                self.logger(f"Error calculating shares for {symbol}: {str(calc_err)}")
                return 0
        
        except Exception as e:
            self.logger(f"Unexpected error calculating shares for {symbol}: {str(e)}")
            return 0
            
    def market_open(self, target_portfolio, current_date):
        """Execute trades to achieve target portfolio"""
        try:
            if not target_portfolio:
                return
                
            price_df = self.data_df_dic[current_date]
            current_value = self.trading.get_portfolio_value()
            
            # Calculate current weights
            current_portfolio = {
                symbol: (pos[1] * pos[2] / current_value)
                for symbol, pos in self.trading.positions.items()
            }
            
            # Track trade details
            trade_details = {
                'buy_value': 0,
                'sell_value': 0,
                'trade_attempts': 0,
                'successful_trades': 0
            }
            
            # Execute trades
            for symbol in set(list(target_portfolio.keys()) + list(current_portfolio.keys())):
                try:
                    # Skip if symbol not in price dataframe
                    if symbol not in price_df.index:
                        self.logger(f"Warning: Symbol {symbol} not found in price data")
                        continue
                    
                    # Get weights
                    target_weight = target_portfolio.get(symbol, 0.0)
                    current_weight = current_portfolio.get(symbol, 0.0)
                    
                    # Determine trade direction
                    if target_weight > current_weight:
                        # Buy
                        weight_diff = target_weight - current_weight
                        shares = self.calculate_shares_from_weight(weight_diff, symbol, price_df)
                        
                        if shares > 0:
                            trade_details['trade_attempts'] += 1
                            # Use price with fallback
                            price = price_df.loc[symbol, 'dlyopen']
                            if pd.isna(price) or price <= 0:
                                price = price_df.loc[symbol, 'dlyprevprc'] if not pd.isna(price_df.loc[symbol, 'dlyprevprc']) else price_df.loc[symbol, 'dlyclose']
                            
                            # Detailed buy logging
                            self.logger(f"BUY ORDER: {symbol}")
                            self.logger(f"  Shares to Buy: {shares}")
                            self.logger(f"  Buy Price: ${price:.2f}")
                            self.logger(f"  Total Buy Value: ${shares * price:,.2f}")
                            
                            # Log current target weight and portfolio context
                            self.logger(f"  Current Weight: {current_weight:.4f}")
                            self.logger(f"  Target Weight: {target_weight:.4f}")
                            
                            if self.trading.order_buy(current_date, symbol, shares, price):
                                trade_details['successful_trades'] += 1
                                trade_details['buy_value'] += shares * price
                                
                                # Log updated position details after buy
                                if symbol in self.trading.positions:
                                    pos = self.trading.positions[symbol]
                                    self.logger(f"  Updated Position: {symbol}")
                                    self.logger(f"    Cost Basis: ${pos[0]:.2f}")
                                    self.logger(f"    Shares: {pos[1]}")
                                    self.logger(f"    Current Price: ${pos[2]:.2f}")
                                    self.logger(f"    Entry Date: {pos[3]}")
                    
                    elif target_weight < current_weight:
                        # Sell
                        weight_diff = current_weight - target_weight
                        shares = self.calculate_shares_from_weight(weight_diff, symbol, price_df)
                        
                        if shares > 0:
                            trade_details['trade_attempts'] += 1
                            # Use price with fallback
                            price = price_df.loc[symbol, 'dlyopen']
                            if pd.isna(price) or price <= 0:
                                price = price_df.loc[symbol, 'dlyprevprc'] if not pd.isna(price_df.loc[symbol, 'dlyprevprc']) else price_df.loc[symbol, 'dlyclose']
                            
                            # Log sell details before selling
                            self.logger(f"SELL ORDER: {symbol}")
                            self.logger(f"  Shares to Sell: {shares}")
                            self.logger(f"  Sell Price: ${price:.2f}")
                            self.logger(f"  Total Sell Value: ${shares * price:,.2f}")
                            
                            # Log current position details before selling
                            if symbol in self.trading.positions:
                                pos = self.trading.positions[symbol]
                                self.logger(f"  Current Position: {symbol}")
                                self.logger(f"    Cost Basis: ${pos[0]:.2f}")
                                self.logger(f"    Shares: {pos[1]}")
                                self.logger(f"    Current Price: ${pos[2]:.2f}")
                                self.logger(f"    Entry Date: {pos[3]}")
                            
                            # Log weight context
                            self.logger(f"  Current Weight: {current_weight:.4f}")
                            self.logger(f"  Target Weight: {target_weight:.4f}")
                            
                            if self.trading.order_sell(current_date, symbol, shares, price):
                                trade_details['successful_trades'] += 1
                                trade_details['sell_value'] += shares * price
                
                except Exception as symbol_trade_err:
                    self.logger(f"Error trading symbol {symbol}: {str(symbol_trade_err)}")
            
            # Log overall trade summary
            self.logger(f"Trade Summary for {current_date}: "
                        f"Attempts: {trade_details['trade_attempts']}, "
                        f"Successful: {trade_details['successful_trades']}, "
                        f"Buy Value: ${trade_details['buy_value']:,.2f}, "
                        f"Sell Value: ${trade_details['sell_value']:,.2f}")
            
            # Update total trade value for turnover calculation
            self.trading.total_trade_value += (trade_details['buy_value'] + trade_details['sell_value'])
        
        except Exception as e:
            self.logger(f"Error in market_open for {current_date}: {str(e)}")
                                        
    def after_market_close(self, current_date):
        """Process end-of-day updates"""
        try:
            # Update prices
            current_data_df = self.trading.update_prices(current_date, self.data_df_dic[current_date], timing='close')
            
            # Calculate margin interest
            self.trading.calculate_margin_interest(current_date)
            
            # Record daily metrics
            portfolio_value = self.trading.get_portfolio_value()
            self.trading.daily_value_list.append(portfolio_value)
            
            # Calculate turnover with more robust method
            # Turnover = (Total Buy + Total Sell) / Portfolio Value
            turnover = (
                100 * self.trading.total_trade_value / portfolio_value
            ) if portfolio_value > 0 else 0
            
            # Reset total trade value after calculating turnover
            total_trade_value_before = self.trading.total_trade_value
            self.trading.total_trade_value = 0
            
            # Store turnover
            self.turnover_dic[current_date] = turnover
            
            # Log daily summary with more context
            self.logger(f"End of Day Summary for {current_date}:")
            self.logger(f"Portfolio Value: ${portfolio_value:,.2f}")
            self.logger(f"Total Trade Value: ${total_trade_value_before:,.2f}")
            self.logger(f"Daily Turnover: {turnover:.2f}%")
        
        except Exception as e:
            self.logger(f"Error in after_market_close for {current_date}: {str(e)}")
        
    def backtest(self):
            try:
                print("Starting backtest...")
                
                for date in tqdm(self.trading_dates[1:]):
                    try:
                        prev_date_idx = self.trading_dates.index(date) - 1
                        prev_date = self.trading_dates[prev_date_idx]
                        prev_data_df = self.data_df_dic[prev_date]
                        target_portfolio = self.before_market_open(date, prev_data_df)
                        #print(target_portfolio)
                        self.market_open(target_portfolio, date)

                        self.after_market_close(date)
                        
                    except Exception as e:
                        self.logger(f"Error on date {date}: {str(e)}")
                        continue
                
                self.save_results()
                print("Backtest completed.")
                return self.trading.daily_value_list
                
            finally:
                if self.txt_file is not None and not self.txt_file.closed:
                    self.txt_file.close()
        
    def save_results(self):
        """Save backtest results"""
        # Save returns
        returns_df = pd.DataFrame({
            'portfolio_value': self.trading.daily_value_list,
            'turnover': self.turnover_dic
        }, index=self.trading_dates)
        returns_df.to_csv(self.return_df_path)
        
        # Close log file
        self.txt_file.close()
    
