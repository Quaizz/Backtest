# Core data processing
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib
import mplcursors
import seaborn as sns
import scipy.stats
import calendar

# Database
import duckdb

# Progress tracking and utilities
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Visualization


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




class BrokersModel(CommissionModel):
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


class FixedSlippageModel:
    """Simple fixed slippage model"""
    def __init__(self, slippage_rate=0.0005):  # Default 0.05% slippage
        """
        Initialize fixed slippage model
        
        Parameters:
        -----------
        slippage_rate : float
            Fixed slippage as percentage of price
        """
        self.slippage_rate = slippage_rate
    
    def calculate_price(self, price: float, is_buy: bool) -> float:
        """
        Calculate price after slippage
        
        Parameters:
        -----------
        price : float
            Original price
        is_buy : bool
            True for buy orders, False for sell orders
        
        Returns:
        --------
        float : Adjusted price after slippage
        """
        if is_buy:
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)


class Trading:
    def __init__(self, cash, commission_model, slippage_model, margin_rate=0.06, is_print=False, txt_file=None):
        """
        Initialize trading system with margin and corporate action support
        """
        self.cash = np.array([cash])
        self.positions = {}  # {symbol: [cost_basis, shares, current_price, entry_date]}
        self.margin_rate = margin_rate
        self.commission_model = commission_model
        self.slippage_model = slippage_model


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
            active_value = 0
            suspended_value = 0
            
            for symbol, pos in self.positions.items():
                cost_basis, shares, current_price, entry_date = pos
                is_suspended = symbol in self.suspended_stocks
                status = "SUSPENDED" if is_suspended else "ACTIVE"
                
                position_value = shares * current_price
                if is_suspended:
                    suspended_value += position_value
                else:
                    active_value += position_value
                    
                position_data.append({
                    'Symbol': symbol,
                    'Status': status,
                    'Cost': cost_basis,
                    'Shares': shares,
                    'Current_Price': current_price,
                    'Change_pct': (current_price/cost_basis - 1) * 100,
                    'Position_Value': position_value,
                    'Position_pct': (position_value / self.daily_value_list[-1] * 100)
                })
            
            position_df = pd.DataFrame(position_data)
            self.logger(position_df.round(2))
            
            if self.suspended_stocks:
                self.logger("\nSuspended Stocks Summary:")
                self.logger(f"Number of Suspended Stocks: {len(self.suspended_stocks)}")
                self.logger(f"Value of Suspended Positions: ${suspended_value:,.2f}")
                total_position_value = active_value + suspended_value
                if total_position_value > 0:
                    self.logger(f"Suspended % of Portfolio: {(suspended_value/total_position_value*100):.2f}%")
                suspended_details = position_df[position_df['Status'] == 'SUSPENDED']
                if not suspended_details.empty:
                    self.logger("\nDetailed Suspended Positions:")
                    self.logger(suspended_details[['Symbol', 'Shares', 'Current_Price', 'Position_Value']].round(2))
        
        total_value = self.get_portfolio_value()
        self.logger(f"\nPortfolio Summary:")
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
        self.logger(f"\n{'-'*15} {date} Delistings {'-'*20}")
        delisted_symbols = []  # Track symbols to be removed
        
        for permno in list(self.positions.keys()):
            if permno not in dsf_data.index:
                continue
                
            if dsf_data.loc[permno, 'dlydelflg'] == 'Y':
                self.logger(f"\nProcessing Delisting for {permno}:")
                # Get delisting return
                dly_ret = dsf_data.loc[permno, 'dlyret']
                if pd.isna(dly_ret):
                    dly_ret = -1.0
                    self.logger(f"  Missing delisting return, using -100%")
                else:
                    self.logger(f"  Delisting return: {dly_ret*100:.2f}%")
                
                # Calculate final value
                shares = self.positions[permno][1]
                last_price = self.positions[permno][2]
                final_value = shares * last_price * (1 + dly_ret)
                
                self.logger(f"  Shares: {shares}")
                self.logger(f"  Last Price: ${last_price:.2f}")
                self.logger(f"  Final Value: ${final_value:,.2f}")
                
                # Add to cash and track for removal
                self.cash[0] += final_value
                delisted_symbols.append(permno)
                
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

        # Remove delisted symbols after processing all
        for symbol in delisted_symbols:
            if symbol in self.positions:
                del self.positions[symbol]
                self.position_count -= 1
                self.logger(f"  Removed {symbol} from positions")
                
            if symbol in self.last_valid_prices:
                del self.last_valid_prices[symbol]




    def handle_events(self, date, dsf_data, distribution_data):
        """Handle all corporate actions before market open"""
        self.logger(f"\n{'-'*15} Processing Corporate Actions for {date} {'-'*15}")
        
        # First handle delistings
        self.handle_delisting(date, dsf_data)
        
        
        
        # Handle corporate actions for remaining positions
        for permno in list(self.positions.keys()):
            if permno not in dsf_data.index:
                self.logger(f"Stock {permno} not found in price data, skipping...")
                continue
                
            dist_flag = dsf_data.loc[permno, 'dlydistretflg']
            
            if dist_flag in ['NO', 'NA', 'T1', 'N1']:
                continue
                
            self.logger(f"\nProcessing corporate action for {permno}:")
            self.logger(f"  Distribution Flag: {dist_flag}")
            
            if dist_flag == 'S2':
                split_factor = dsf_data.loc[permno, 'dlyfacprc']
                self.logger(f"  Stock Split detected. Factor: {split_factor}")
                self.handle_split(permno, split_factor)
                continue

            events = distribution_data[
                (distribution_data['permno'] == permno) & 
                (distribution_data['disexdt'] == date)
            ]


            if not events.empty:
                self.logger(f"  Found {len(events)} distribution events")
                # Sort events by sequence number
                sorted_events = events.sort_values('disseqnbr')
                self.logger(f"  Processing events in sequence:")
                
                for _, event in sorted_events.iterrows():
                    self.logger(f"    Processing event sequence number: {event['disseqnbr']}")
                    self.process_distribution_event(permno, event, date)

        # Process any pending payments due today
        self.process_pending_payments(date)

    def process_distribution_event(self, permno, event, current_date):
        """Process a single distribution event"""
        self.logger(f"\nProcessing Distribution Event for {permno}:")
        self.logger(f"  Event Type: {event['distype']}")
        self.logger(f"  Detail Type: {event.get('disdetailtype', 'N/A')}")
        
        if event['distype'] in ['SP', 'SD', 'CD', 'ROC']:

            # Handle NaN dividend amount
            div_amount = event['disdivamt']
            if pd.isna(div_amount):
                self.logger(f"  Warning: NaN dividend amount, setting to 0")
                div_amount = 0.0

            raw_cash = div_amount * self.positions[permno][1]
            tax_rate = self.get_tax_rate(
                event['distaxtype'],
                entry_date=self.positions[permno][3],  # Entry date from position
                current_date=current_date
            )
            net_cash = raw_cash * (1 - tax_rate)
            
            self.logger(f"  Cash Distribution:")
            self.logger(f"    Amount per share: ${div_amount:.4f}")
            self.logger(f"    Tax Rate: {tax_rate*100:.1f}%")
            self.logger(f"    Gross Amount: ${raw_cash:,.2f}")
            self.logger(f"    Net Amount: ${net_cash:,.2f}")
            
            pay_date = event['dispaydt']
            # If pay_date is today or past, process immediately
            if pay_date <= current_date:
                self.logger(f"    Processing payment immediately")
                self.cash[0] += net_cash

                # Adjust cost basis for Return of Capital
                if event['distype'] in ['ROC', 'SP'] and not pd.isna(div_amount):
                    old_cost_basis = self.positions[permno][0]
                    new_cost_basis = max(0.01, old_cost_basis - div_amount)
                    self.positions[permno][0] = new_cost_basis
                    self.logger(f"    {event['distype']} - Adjusted cost basis from ${old_cost_basis:.4f} to ${new_cost_basis:.4f}")
            else:
                if pay_date not in self.pending_cash:
                    self.pending_cash[pay_date] = []
                self.pending_cash[pay_date].append((net_cash, permno, event['distype']))
                self.logger(f"    Scheduled for payment on: {pay_date}")
            
        elif event['distype'] == 'FRS':
            if event['disdetailtype'] == 'STKSPL':
                factor = 1 + event['disfacshr']
                self.logger(f"  Stock Split:")
                self.logger(f"    Split Factor: {factor}")
                self.handle_split(permno, factor)
                
            elif event['disdetailtype'] == 'STKDIV':
                shares = self.positions[permno][1]
                new_shares = int(shares * event['disfacshr'])
                pay_date = event['dispaydt']
                
                self.logger(f"  Stock Dividend:")
                self.logger(f"    Current Shares: {shares}")
                self.logger(f"    New Shares: {new_shares}")
                self.logger(f"    Payment Date: {pay_date}")
                
                # If pay_date is today or past, process immediately
                if pay_date <= current_date:
                    self.logger(f"    Processing stock dividend immediately")
                    self.positions[permno][1] += new_shares
                else:
                    if pay_date not in self.pending_shares:
                        self.pending_shares[pay_date] = []
                    self.pending_shares[pay_date].append((permno, new_shares))
                    self.logger(f"    Scheduled for payment on: {pay_date}")



    def handle_split(self, permno, factor):
        """Handle stock split"""
        if permno in self.positions:
            self.logger(f"\nProcessing Split for {permno}:")
            position = self.positions[permno]
            
            self.logger(f"  Before Split:")
            self.logger(f"    Shares: {position[1]}")
            self.logger(f"    Cost Basis: ${position[0]:.4f}")
            self.logger(f"    Current Price: ${position[2]:.4f}")
            
            position[1] = int(position[1] * factor)  # Round to integer shares,  Adjust shares
            position[0] /= factor  # Adjust cost basis
            position[2] /= factor  # Adjust current price
            
            self.logger(f"  After Split:")
            self.logger(f"    Shares: {position[1]}")
            self.logger(f"    Cost Basis: ${position[0]:.4f}")
            self.logger(f"    Current Price: ${position[2]:.4f}")
            
            if permno in self.last_valid_prices:
                old_price = self.last_valid_prices[permno]
                self.last_valid_prices[permno] /= factor
                self.logger(f"  Updated last valid price from ${old_price:.4f} to ${self.last_valid_prices[permno]:.4f}")


    def process_pending_payments(self, date):
        """Process pending cash and share payments"""
        # Process cash dividends
        if date in self.pending_cash:
            self.logger(f"\nProcessing pending cash payments for {date}:")
            for amount, source_permno, dist_type  in self.pending_cash[date]:
                self.cash[0] += amount
                self.logger(f"  Received ${amount:,.2f} from stock {source_permno}")

                # Adjust cost basis for Return of Capital
                if dist_type in ['ROC', 'SP'] and source_permno in self.positions:
                    old_cost_basis = self.positions[source_permno][0]
                    shares = self.positions[source_permno][1]
                    amount_per_share = amount / shares
                    new_cost_basis = max(0.01, old_cost_basis - amount_per_share)
                    self.positions[source_permno][0] = new_cost_basis
                    self.logger(f"    {dist_type} - Adjusted cost basis from ${old_cost_basis:.4f} to ${new_cost_basis:.4f}")

            del self.pending_cash[date]
        
        # Process stock dividends - only if still holding the stock
        if date in self.pending_shares:
            self.logger(f"\nProcessing pending share payments for {date}:")
            for permno, new_shares in self.pending_shares[date]:
                if permno in self.positions:
                    current_shares = self.positions[permno][1]
                    current_cost_basis = self.positions[permno][0]
                    
                    # Calculate new cost basis (maintain same total cost)
                    total_cost = current_cost_basis * current_shares
                    new_total_shares = current_shares + new_shares
                    new_cost_basis = total_cost / new_total_shares
                    
                    # Update position
                    self.positions[permno][0] = new_cost_basis  # Update cost basis
                    self.positions[permno][1] = new_total_shares  # Update shares
                    
                    self.logger(f"  Added {new_shares} shares to {permno}")
                    self.logger(f"  Adjusted cost basis from ${current_cost_basis:.2f} to ${new_cost_basis:.2f}")
                else:
                    self.logger(f"  Skipped {new_shares} shares for {permno} - no longer holding position")
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
            
        execution_price = self.slippage_model.calculate_price(price, is_buy=True)
        total_cost = shares * execution_price
        commission = self.commission_model.calculate_commission(shares, execution_price)
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
        'execution_price': execution_price,
        'slippage_cost': (execution_price - price) * shares,
        'commission': commission,
        'margin_used': max(0, -self.cash[0])
        })
        
        return True


    def order_sell(self, date, symbol, shares, price):
        """Place sell order with margin paydown and tax calculation"""
        if symbol not in self.positions or pd.isna(price) or symbol in self.suspended_stocks:
            return False
            
        current_shares = self.positions[symbol][1]
        if shares > current_shares:
            shares = current_shares
        
        execution_price = self.slippage_model.calculate_price(price, is_buy=False)
        # Calculate commission
        commission = self.commission_model.calculate_commission(shares, execution_price)
        
        # Calculate capital gains
        cost_basis = self.positions[symbol][0]
        entry_date = self.positions[symbol][3]
        gain_per_share = price - cost_basis
        total_gain = gain_per_share * shares
        
        # Calculate tax
        tax_rate = self.get_tax_rate('C', entry_date, date)  # 'C' for Capital Gains
        tax_amount = max(0, total_gain * tax_rate)  # Only tax positive gains
        
        # Calculate final proceeds
        gross_proceeds = shares * price
        proceeds = gross_proceeds - commission - tax_amount
        self.total_trade_value += gross_proceeds
        
        self.logger(f"\nSell Order Tax Calculation for {symbol}:")
        self.logger(f"  Cost Basis: ${cost_basis:.2f}")
        self.logger(f"  Sell Price: ${price:.2f}")
        self.logger(f"  Gain per Share: ${gain_per_share:.2f}")
        self.logger(f"  Total Gain: ${total_gain:.2f}")
        self.logger(f"  Tax Rate: {tax_rate*100:.1f}%")
        self.logger(f"  Tax Amount: ${tax_amount:.2f}")
        self.logger(f"  Commission: ${commission:.2f}")
        self.logger(f"  Net Proceeds: ${proceeds:.2f}")
        
        # Update position
        if shares == current_shares:
            del self.positions[symbol]
            self.position_count -= 1
        else:
            self.positions[symbol][1] -= shares
        
        # Update cash and margin
        self.cash[0] += proceeds
        self.repay_margin_loans(proceeds, date)
        
        self.last_valid_prices[symbol] = price
        
        # Rest of the order_sell logic...
        self.trades_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'execution_price': execution_price,
            'slippage_cost': (price - execution_price) * shares,
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
        
        # Instead of clearing, create a new set to track currently suspended stocks
        currently_suspended = set()
        
        # Tracking variables
        total_stocks = len(self.positions)
        imputed_stocks = 0
        suspended_stocks = 0
        
        for symbol in list(self.positions.keys()):
            if symbol not in price_data.index:
                suspended_stocks += 1
                currently_suspended.add(symbol)
                self.logger(f"  Not found in price data - marked as suspended")
                continue
            
            stock_data = price_data.loc[symbol]
            price_flag = stock_data[flag_field]
            
            # Check for suspended or halted stocks
            if price_flag in ['HA', 'SU', 'MP']:
                currently_suspended.add(symbol)
                suspended_stocks += 1
                #self.logger(f"  Status flag: {price_flag} - marked as suspended")
                
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
                    currently_suspended.add(symbol)
                    suspended_stocks += 1
                    continue
            else:
                # Update position and last valid prices with the current price
                self.positions[symbol][2] = price_data.loc[symbol, price_field]
                self.last_valid_prices[symbol] = price_data.loc[symbol, price_field]
                
                # If stock has valid price, remove from suspended list if it was there
                if symbol in self.suspended_stocks:
                    self.suspended_stocks.remove(symbol)
        
        # Update suspended_stocks with currently suspended stocks
        self.suspended_stocks.update(currently_suspended)
        
        return price_data


    def get_portfolio_value(self, include_margin_cost=True):
        """Calculate total portfolio value"""
        value = self.cash[0]
        for symbol, pos in self.positions.items():
            value += pos[1] * pos[2]
            
        if include_margin_cost:
            value -= sum(self.margin_loans.values())
            
        return value

    def get_tax_rate(self, tax_type, entry_date=None, current_date=None):
        """
        Get tax rate based on distribution tax type and holding period
        
        Args:
            tax_type: Tax status type from CRSP (C, D, F, G, N, N/A, P, R, T, U, X)
            entry_date: Entry date of the position (for determining holding period)
            current_date: Current date for tax calculation
            
        Returns:
            float: Tax rate to apply
        """
        #self.logger(f"Getting tax rate for distribution type: {tax_type}")
        
        # Check if it's a short-term holding
        is_short_term = False
        if entry_date and current_date:
            holding_period = pd.Timestamp(current_date) - pd.Timestamp(entry_date)
            is_short_term = holding_period.days < 365
        #    self.logger(f"Holding period: {holding_period.days} days")
        
        # Base tax rates
        if is_short_term:
            tax_rates = {
                'C': 0.40,    # Short-term Capital Gains
                'D': 0.40,    # Short-term Dividend
                'F': 0.40,    # Full
                'G': 0.40,    # Short-term Gain/Loss
                'N': 0.0,     # Non-Taxable
                'N/A': 0.40,  # Not Applicable
                'P': 0.40,    # Plan
                'R': 0.0,     # Return of Capital - not taxed but reduces cost basis
                'T': 0.40,    # Tax Receipt
                'U': 0.40,    # Unspecified
                'X': 0.40     # Unknown
            }
        else:
            tax_rates = {
                'C': 0.20,    # Long-term Capital Gains
                'D': 0.25,    # Long-term Dividend
                'F': 0.25,    # Full
                'G': 0.20,    # Long-term Gain/Loss
                'N': 0.0,     # Non-Taxable
                'N/A': 0.25,  # Not Applicable
                'P': 0.25,    # Plan
                'R': 0.0,     # Return of Capital
                'T': 0.25,    # Tax Receipt
                'U': 0.25,    # Unspecified
                'X': 0.25     # Unknown
            }
        
        rate = tax_rates.get(tax_type, 0.40 if is_short_term else 0.25)  # Default based on term
        #self.logger(f"Applied tax rate: {rate*100:.1f}% ({'Short-term' if is_short_term else 'Long-term'})")
        
        return 0.0
    
    def get_total_position_value(self):
        """Calculate total value of current positions"""
        total_value = 0
        for symbol, pos in self.positions.items():
            total_value += pos[1] * pos[2]  # shares * current price
        return total_value



class Backtest:
    def __init__(self, cash, commission_model, slippage_model, data_df_dic, trading_dates, 
                 distribution_data, factor_list, dir_='results', save_format='html',
                 stock_num=50, margin_rate=0.06, rebalance_freq='1d', 
                 rebalance_day=1, min_price=2.0, market_cap_percentile=0.05,
                 buy_and_hold_list=None,gvkeyx='000003', weight_change_threshold = 0.0, strategy=None):
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
        self.dir_ = dir_
        self.save_format = save_format

        self.png_save_path = os.path.join(dir_, f'portfolio_return.{save_format}')
        self.return_df_path = os.path.join(dir_, 'return_df.csv')

        # Store data
        self.data_df_dic = data_df_dic
        self.trading_dates = trading_dates
        self.distribution_data = distribution_data
        self.total_trade_value = 0
        self.factor_list = factor_list if isinstance(factor_list, list) else [factor_list]
        

        self.daily_trade_details = {}  # To store daily trading statistics
        # Performance tracking
        self.turnover_dic = {}

        # Initialize backtest logging
        self.txt_file = open(os.path.join(dir_, 'backtest_logs.txt'), 'w+')


        # Store strategy parameters
        self.stock_num = stock_num
        self.margin_rate = margin_rate
        self.rebalance_freq = rebalance_freq
        self.rebalance_day = rebalance_day
        self.min_price = min_price
        self.market_cap_percentile = market_cap_percentile

        self.rebalance_dates = self.get_rebalance_dates()
        # Load benchmark and risk-free data
        self.gvkeyx = gvkeyx
        self._load_benchmark_data()
        self.weight_change_threshold = weight_change_threshold
        trading_log_path = os.path.join(dir_, 'trading_logs.txt')
        # Initialize slippage model
        
        # Initialize trading system with logging
        self.trading = Trading(
            cash=cash, 
            commission_model=commission_model, 
            slippage_model=slippage_model,
            margin_rate=self.margin_rate,
            is_print=False,  # This will print to console
            txt_file=open(trading_log_path, 'w+')  # This will save to file
        )

        self.backtest_data = {
        'portfolio_values': [],      # For returns and drawdown calculation
        'benchmark_values': [],      # For benchmark comparison
        'turnover': {},             # For turnover analysis
        'trades_count': 0,          # Total number of trades
        'total_trade_value': 0      # For turnover calculation
        }

        # Store buy and hold portfolio if provided
        self.buy_and_hold_list = buy_and_hold_list
        self.strategy = strategy


        if buy_and_hold_list is not None:
            self.logger(f"Using buy and hold strategy with {len(buy_and_hold_list)} stocks")
            # Verify weights sum to approximately 1
            total_weight = sum(buy_and_hold_list.values())
            if not 0.99 <= total_weight <= 1.01:
                self.logger(f"Warning: Buy and hold weights sum to {total_weight}, normalizing...")
                self.buy_and_hold_list = {
                    symbol: weight/total_weight 
                    for symbol, weight in buy_and_hold_list.items()
                }


        
    def logger(self, txt_str):
        """Log message to file"""
        if self.txt_file is not None and not self.txt_file.closed:
            print(txt_str, file=self.txt_file)
        #print(txt_str)  # Also print to console
            
    def get_rebalance_dates(self):
        """Get rebalancing dates based on frequency"""
        if self.rebalance_freq == '1d':
            return self.trading_dates
        
        # If not daily rebalancing, convert dates to DataFrame
        dates_df = pd.DataFrame({
            'date': pd.to_datetime(self.trading_dates)
        })
        
        if self.rebalance_freq == '1w':
            mask = dates_df['date'].dt.weekday == self.rebalance_day
            # Return datetime objects to match the format expected elsewhere
            return dates_df[mask]['date'].tolist()
        
        elif self.rebalance_freq == '1m':
            dates_df['ym'] = dates_df['date'].dt.strftime('%Y%m')
            monthly_groups = dates_df.groupby('ym')
            
            rebalance_dates = []
            for _, group in monthly_groups:
                group = group.sort_values('date')
                target_day = min(self.rebalance_day, group['date'].iloc[-1].day)
                valid_dates = group[group['date'].dt.day <= target_day]
                
                if not valid_dates.empty:
                    rebalance_dates.append(valid_dates.iloc[-1]['date'])
                else:
                    rebalance_dates.append(group.iloc[0]['date'])
            
            # Return datetime objects to match the format expected elsewhere
            return rebalance_dates
        
        # Fall back to daily rebalancing if frequency not recognized
        return self.trading_dates
        
    def _load_benchmark_data(self):
        """
        Load benchmark data and initialize benchmark-related class members.
        """
        # Load benchmark data
        benchmark_file = Path(f'Data_all/Benchmark_data/{self.gvkeyx}/benchmark_{self.gvkeyx}.parquet')
        if not benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
            
        # Read benchmark data
        benchmark_df = pd.read_parquet(benchmark_file)
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        benchmark_df.set_index('date', inplace=True)
        
        # Get benchmark returns and prices for our trading dates
        trading_dates = pd.to_datetime(self.trading_dates)[1:]
        self.benchmark_returns = benchmark_df.loc[trading_dates, 'ret_idx']
        self.benchmark_prices = benchmark_df.loc[trading_dates, 'tr_idx']
        
        # Calculate cumulative returns relative to first date's price
        base_price = self.benchmark_prices.iloc[0]
        self.benchmark_cum = self.benchmark_prices / base_price
        
        # Load risk-free rates
        rf_file = Path('Data_all/RF_data/rf_rates.parquet')
        if not rf_file.exists():
            raise FileNotFoundError("Risk-free rate file not found")
            
        rf_df = pd.read_parquet(rf_file)
        rf_df['date'] = pd.to_datetime(rf_df['date'])
        rf_df.set_index('date', inplace=True)
        
        # Get risk-free rates for our trading dates
        self.daily_rf_rates = rf_df.loc[trading_dates, 'rf']   



    def before_market_open(self, date, prev_data_df, current_data_df):
        """Process events and generate targets before market open"""
        try:
            # Handle delistings and corporate actions
            self.trading.handle_events(
                date, 
                current_data_df, 
                self.distribution_data[self.distribution_data['disexdt'] == date]
            )
            
            # Generate new portfolio targets on rebalance dates
            if date in self.rebalance_dates:
                return self.generate_portfolio_targets(date, prev_data_df)
            return {}
        except Exception as e:
            self.logger(f"Error in before_market_open for {date}: {str(e)}")
            return {}
    
    def generate_portfolio_targets(self, date, data_df):
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
            # If external strategy is provided, delegate to it
            if self.strategy is not None:
                # Check if the strategy has the expected method
                if hasattr(self.strategy, 'generate_portfolio_targets'):

                    
                    return self.strategy.generate_portfolio_targets(
                        date,
                        data_df, 
                        min_price=self.min_price,
                        market_cap_percentile=self.market_cap_percentile,
                        stock_num=self.stock_num,
                        logger=self.logger
                    )

                    
                else:
                    self.logger("Error: Strategy object does not implement generate_portfolio_targets method")
                    return {}
                
            # If buy and hold list is provided, use it
            if self.buy_and_hold_list is not None:
                # Only return stocks that exist in current universe
                available_stocks = {
                    symbol: weight 
                    for symbol, weight in self.buy_and_hold_list.items()
                    if symbol in data_df.index
                }
                
                if len(available_stocks) < len(self.buy_and_hold_list):
                    self.logger(f"Warning: Only {len(available_stocks)} of {len(self.buy_and_hold_list)} "
                              f"buy and hold stocks available")
                    
                    # Renormalize weights if some stocks are unavailable
                    total_weight = sum(available_stocks.values())
                    if total_weight > 0:
                        available_stocks = {
                            symbol: weight/total_weight 
                            for symbol, weight in available_stocks.items()
                        }
                
                return available_stocks
            # Validate data
            if data_df is None or len(data_df) == 0:
                self.logger("Error: Empty dataframe passed to generate_portfolio_targets")
                return {}

            # Filter universe
            
            universe_df = data_df[
                (data_df['dlyprc'] > self.min_price) &
                (data_df['dlycap'] >  data_df['dlycap'].quantile(self.market_cap_percentile))
            ].copy()
            

            # Calculate composite rank for each factor
            ranks = pd.DataFrame(index=universe_df.index)
            valid_factors = []

            for factor in self.factor_list:
                
                # Additional check for NaN values
                factor_series = universe_df[factor]
                if factor_series.isna().all():
                    self.logger(f"Warning: All values for factor {factor} are NaN")
                    continue
                
                # Calculate percentile rank, dropping NaNs
                rank_series = factor_series.rank(ascending=True, na_option='top')

                ranks[f"{factor}_rank"] = rank_series
                valid_factors.append(factor)
            

            
            # Calculate composite score
            try:
                ranks['composite_score'] = ranks.sum(axis=1)
            except Exception as e:
                self.logger(f"Error calculating composite score: {str(e)}")
                return {}
            
            # Remove rows with NaN composite score
            universe_df['composite_score'] = ranks['composite_score']
            universe_df = universe_df.dropna(subset=['composite_score'])
            

            
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


    def market_open(self, target_portfolio, current_date, current_data_df ):
        """Execute trades to achieve target portfolio"""
        
        if not target_portfolio:
            return
            
        #price_df = self.data_df_dic[current_date]
        price_df = self.trading.update_prices(current_date, current_data_df, timing='open')

        current_value = self.trading.get_portfolio_value()
        
        # Filter out delisted stocks from target portfolio
        delisted_symbols = [
            symbol for symbol in target_portfolio.keys()
            if symbol in price_df.index and price_df.loc[symbol, 'dlydelflg'] == 'Y'
        ]
        for symbol in delisted_symbols:
            if symbol in target_portfolio:
                self.logger(f"Removing delisted stock {symbol} from target portfolio")
                del target_portfolio[symbol]


        # Calculate current weights
        current_portfolio = {
            symbol: (pos[1] * pos[2] / current_value)
            for symbol, pos in self.trading.positions.items()
        }
        #print(current_portfolio)
        # Track trade details
        trade_details = {
            'buy_value': 0,
            'sell_value': 0,
            'trade_attempts': 0,
            'successful_trades': 0
        }
        
        # 1. Sell positions that are not in target portfolio
        for symbol in list(current_portfolio.keys()):
            if symbol not in target_portfolio:
                if symbol in self.trading.positions:
                    shares = self.trading.positions[symbol][1]
                    
                    # Check if symbol is in price_df
                    if symbol in price_df.index:
                        price = price_df.loc[symbol, 'dlyopen']
                        self.logger(f"Liquidating position in {symbol}: {shares} shares at price ${price:.2f}")
                    else:
                        # Use last valid price if not in current price data
                        if symbol in self.trading.last_valid_prices:
                            price = self.trading.last_valid_prices[symbol]
                            self.logger(f"Liquidating position in {symbol} using last valid price: {shares} shares at ${price:.2f}")
                        else:
                            self.logger(f"Cannot liquidate {symbol} - no valid price available")
                            continue
                    
                    if self.trading.order_sell(current_date, symbol, shares, price):
                        trade_details['successful_trades'] += 1
                        trade_details['sell_value'] += shares * price
                    trade_details['trade_attempts'] += 1

        # 2. Calculate weight changes for existing positions
        weight_changes = {}
        for symbol in set(target_portfolio.keys()) & set(current_portfolio.keys()):
            weight_diff = target_portfolio[symbol] - current_portfolio[symbol]
            if abs(weight_diff) > self.weight_change_threshold:  # Small threshold to avoid tiny trades
                weight_changes[symbol] = weight_diff

        # 3. Execute sells for reducing positions
        for symbol, weight_diff in weight_changes.items():
            if weight_diff < 0:
                shares = self.calculate_shares_from_weight(abs(weight_diff), symbol, price_df)
                if shares > 0:
                    price = price_df.loc[symbol, 'dlyopen']
                    trade_details['trade_attempts'] += 1
                    if self.trading.order_sell(current_date, symbol, shares, price):
                        trade_details['successful_trades'] += 1
                        trade_details['sell_value'] += shares * price

        # 4. Execute buys for increasing positions
        for symbol, weight_diff in weight_changes.items():
            if weight_diff > 0:
                shares = self.calculate_shares_from_weight(weight_diff, symbol, price_df)
                if shares > 0:
                    price = price_df.loc[symbol, 'dlyopen']
                    trade_details['trade_attempts'] += 1
                    if self.trading.order_buy(current_date, symbol, shares, price):
                        trade_details['successful_trades'] += 1
                        trade_details['buy_value'] += shares * price

        # 5. Buy new positions
        for symbol in set(target_portfolio.keys()) - set(current_portfolio.keys()):
            if not price_df.loc[symbol, 'tradingstatusflg'] in ['H', 'S']:  # Skip suspended stocks
                shares = self.calculate_shares_from_weight(target_portfolio[symbol], symbol, price_df)
                if shares > 0:
                    price = price_df.loc[symbol, 'dlyopen']
                    trade_details['trade_attempts'] += 1
                    if self.trading.order_buy(current_date, symbol, shares, price):
                        trade_details['successful_trades'] += 1
                        trade_details['buy_value'] += shares * price

        
        # Print trade and position information using Trading class methods
        self.trading.print_trading_info(current_date)
        self.trading.print_position_info(current_date, price_df)
        
        # Update total trade value for turnover calculation
        self.trading.total_trade_value += (trade_details['buy_value'] + trade_details['sell_value'])
        # Store daily trade details
        self.daily_trade_details[current_date] = trade_details
        self.logger(f"\nDaily Trading Statistics:")
        self.logger(f"  Trade Attempts: {trade_details['trade_attempts']}")
        self.logger(f"  Successful Trades: {trade_details['successful_trades']}")
        self.logger(f"  Buy Value: ${trade_details['buy_value']:,.2f}")
        self.logger(f"  Sell Value: ${trade_details['sell_value']:,.2f}")


        

    def after_market_close(self, current_date, current_data_df):
        """Process end-of-day updates"""
        try:
            # Update prices
            _ = self.trading.update_prices(current_date, current_data_df, timing='close')
            
            # Calculate margin interest
            self.trading.calculate_margin_interest(current_date)
            
            # Record daily metrics
            portfolio_value = self.trading.get_portfolio_value()
            self.trading.daily_value_list.append(portfolio_value)
            
            # Calculate turnover with more robust method
            # Turnover = (Total Buy + Total Sell) / Portfolio Value
            turnover = (
                100 * self.trading.total_trade_value /2 / portfolio_value
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


            # Reset trading system
            self.trading.daily_value_list = []
            self.trading.trades_history = []
            self.trading.positions = {}
            self.trading.margin_loans = {}
            self.trading.margin_interest = []
            self.turnover_dic = {}
            # Reset backtest data
            self.backtest_data = {
                'portfolio_values': [],
                'benchmark_values': [],
                'turnover': {},
                'trades_count': 0,
                'total_trade_value': 0
            }

            if self.txt_file is None or self.txt_file.closed:
                self.txt_file = open(os.path.join(self.dir_, 'backtest_logs.txt'), 'w+')
            if self.trading.txt_file is None or self.trading.txt_file.closed:
                self.trading.txt_file = open(os.path.join(self.dir_, 'trading_logs.txt'), 'w+')

            # Clean log files
            if self.trading.txt_file is not None:
                self.trading.txt_file.seek(0)
                self.trading.txt_file.truncate()
            if self.txt_file is not None:
                self.txt_file.seek(0)
                self.txt_file.truncate()

            # Store initial values
            initial_value = self.trading.cash[0]
            self.trading.daily_value_list.append(initial_value)
            self.backtest_data['portfolio_values'].append(initial_value)
            self.backtest_data['benchmark_values'].append(self.benchmark_prices.iloc[0])

            for date in tqdm(self.trading_dates[1:]):
                
                prev_date_idx = self.trading_dates.index(date) - 1
                prev_date = self.trading_dates[prev_date_idx]
                prev_data_df = self.data_df_dic[prev_date]
                prev_data_df = prev_data_df[prev_data_df['tradingstatusflg'] == 'A']


                current_date_idx = self.trading_dates.index(date)
                current_date = self.trading_dates[current_date_idx]
                current_data_df = self.data_df_dic[current_date]
                current_data_df = current_data_df[current_data_df['tradingstatusflg'] != 'X']

                target_portfolio = self.before_market_open(date, prev_data_df,current_data_df)
                #print(target_portfolio)
                self.market_open(target_portfolio, date, current_data_df)

                self.after_market_close(date, current_data_df)
                # Record essential data
                self.backtest_data['portfolio_values'].append(self.trading.daily_value_list[-1])
                self.backtest_data['benchmark_values'].append(self.benchmark_prices.loc[date])
                if date in self.turnover_dic:
                    self.backtest_data['turnover'][date] = self.turnover_dic[date]

            self.backtest_data['trades_count'] = len(self.trading.trades_history)
            self.backtest_data['total_trade_value'] = self.trading.total_trade_value          


            self.save_results()
            print("Backtest completed.")
            #return self.trading.daily_value_list
            
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
        
        # Close Trading class log file
        if self.trading.txt_file is not None:
            self.trading.txt_file.close()




    def calculate_performance_metrics(self):
        """Calculate all performance metrics from raw data and store results"""
        # Convert raw data to numpy arrays
        portfolio_values = np.array(self.backtest_data['portfolio_values'])
        benchmark_values = np.array(self.backtest_data['benchmark_values'])
        
        # Store processed data for plotting
        self.processed_data = {
            'dates': self.trading_dates[1:],  # Trading dates excluding first date
            'portfolio_returns': portfolio_values[1:] / portfolio_values[:-1] - 1,
            'benchmark_returns': benchmark_values[1:] / benchmark_values[:-1] - 1,
            'portfolio_cum_returns': portfolio_values[1:] / portfolio_values[0],  # Cumulative returns starting from t=1
            'benchmark_cum_returns': benchmark_values[1:] / benchmark_values[0],  # Cumulative returns starting from t=1
            'portfolio_drawdown': None,
            'benchmark_drawdown': None,
            'excess_returns': None,
            'turnover': list(self.backtest_data['turnover'].values())
        }
        
        # Calculate excess returns vs benchmark
        self.processed_data['excess_returns'] = (
            self.processed_data['portfolio_returns'] - 
            self.processed_data['benchmark_returns']
        )
        
        # Calculate excess returns vs risk-free rate
        excess_returns_rf = self.processed_data['portfolio_returns'] - self.daily_rf_rates.values
        
        # Calculate drawdowns
        portfolio_peak = np.maximum.accumulate(portfolio_values[1:])  # Start from t=1
        benchmark_peak = np.maximum.accumulate(benchmark_values[1:])  # Start from t=1
        self.processed_data['portfolio_drawdown'] = (
            (portfolio_values[1:] - portfolio_peak) / portfolio_peak * 100
        )
        self.processed_data['benchmark_drawdown'] = (
            (benchmark_values[1:] - benchmark_peak) / benchmark_peak * 100
        )
        
        # Calculate metrics
        years = (len(self.trading_dates) - 1) / 252
        
        # Store calculated metrics
        self.metrics = {
            'Total Return (%)': (portfolio_values[-1]/portfolio_values[0] - 1) * 100,
            'Annualized Return (%)': ((portfolio_values[-1]/portfolio_values[0])**(1/years) - 1) * 100,
            'Benchmark Total Return (%)': (benchmark_values[-1]/benchmark_values[0] - 1) * 100,
            'Benchmark Annualized Return (%)': ((benchmark_values[-1]/benchmark_values[0])**(1/years) - 1) * 100,
            'Maximum Drawdown (%)': np.min(self.processed_data['portfolio_drawdown']),
            'Sharpe Ratio': (np.mean(excess_returns_rf) / 
                            np.std(excess_returns_rf) * np.sqrt(252)),
            'Std of Excess Return (%)': np.std(self.processed_data['excess_returns']) * np.sqrt(252) * 100,
            'Information Ratio': (np.mean(self.processed_data['excess_returns']) * 
                                np.sqrt(252) / np.std(self.processed_data['excess_returns'])),
            'Total Trade Value': self.backtest_data['total_trade_value'],
            'Avg Daily Turnover (%)': np.mean(self.processed_data['turnover']),
            'Number of Trades': self.backtest_data['trades_count']
        }
        
        return pd.DataFrame.from_dict(self.metrics, orient='index', columns=['Value'])
    

    def plot_cumulative_returns(self, ax=None, show_benchmark=True, show_excess=True, interactive=True):
        """
        Plot cumulative returns with interactive features.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        show_benchmark : bool, optional
            Whether to show benchmark returns
        show_excess : bool, optional
            Whether to show excess returns
        interactive : bool, optional
            Whether to enable interactive features
            
        Returns:
        --------
        matplotlib.axes.Axes
            The plot axes
        """
        if not hasattr(self, 'processed_data'):
            self.calculate_performance_metrics()

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        lines = []
        
        # Calculate relative returns starting from 0%
        initial_portfolio_value = self.processed_data['portfolio_cum_returns'][0]
        initial_benchmark_value = self.processed_data['benchmark_cum_returns'][0]
        
        relative_portfolio_returns = self.processed_data['portfolio_cum_returns'] / initial_portfolio_value -1
        relative_benchmark_returns = self.processed_data['benchmark_cum_returns'] / initial_benchmark_value -1
        
        # Portfolio line
        portfolio_line = ax.plot(
            self.processed_data['dates'],
            relative_portfolio_returns,
            label=f'Portfolio (Ann. Ret: {self.metrics["Annualized Return (%)"]:.1f}%)',
            color='blue',
            linewidth=2
        )[0]
        lines.append(portfolio_line)

        # Benchmark line
        if show_benchmark:
            benchmark_line = ax.plot(
                self.processed_data['dates'],
                relative_benchmark_returns,
                label=f'Benchmark (Ann. Ret: {self.metrics["Benchmark Annualized Return (%)"]:.1f}%)',
                color='red',
                linestyle='--',
                linewidth=1.5
            )[0]
            lines.append(benchmark_line)

        # Excess returns
        if show_excess and show_benchmark:
            excess_returns = relative_portfolio_returns - relative_benchmark_returns
            excess_line = ax.plot(
                self.processed_data['dates'],
                excess_returns,
                label=f'Excess Returns (IR: {self.metrics["Information Ratio"]:.2f})',
                color='green',
                linestyle=':',
                linewidth=1
            )[0]
            lines.append(excess_line)

        # Customize plot
        ax.set_title('Cumulative Returns', fontsize=12, pad=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Return', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Add interactive features
        if interactive:
            cursor = mplcursors.cursor(lines, hover=True)
            
            @cursor.connect("add")
            def on_hover(sel):
                x, y = sel.target
                date_idx = np.abs(matplotlib.dates.date2num(self.processed_data['dates']) - x).argmin()
                date_str = self.processed_data['dates'][date_idx].strftime('%Y-%m-%d')
                
                if sel.artist == portfolio_line:
                    # Calculate rolling metrics
                    window = 20  # 20-day window
                    if date_idx >= window:
                        rolling_return = (np.prod(1 + self.processed_data['portfolio_returns']
                                        [date_idx-window:date_idx]) - 1) * 100
                        rolling_vol = np.std(self.processed_data['portfolio_returns']
                                        [date_idx-window:date_idx]) * np.sqrt(252) * 100
                    else:
                        rolling_return = (np.prod(1 + self.processed_data['portfolio_returns']
                                        [:date_idx+1]) - 1) * 100
                        rolling_vol = np.std(self.processed_data['portfolio_returns']
                                        [:date_idx+1]) * np.sqrt(252) * 100

                    daily_return = self.processed_data['portfolio_returns'][date_idx] * 100
                    cum_return = (y ) * 100
                    
                    sel.annotation.set_text(
                        f"Portfolio\nDate: {date_str}\n"
                        f"Daily Return: {daily_return:.2f}%\n"
                        f"Cumulative Return: {cum_return:.2f}%\n"
                        f"20D Return: {rolling_return:.2f}%\n"
                        f"20D Vol (Ann.): {rolling_vol:.2f}%"
                    )
                
                elif sel.artist == benchmark_line:
                    daily_return = self.processed_data['benchmark_returns'][date_idx] * 100
                    cum_return = (y ) * 100
                    
                    # Calculate rolling benchmark metrics
                    window = 20
                    if date_idx >= window:
                        rolling_return = (np.prod(1 + self.processed_data['benchmark_returns']
                                        [date_idx-window:date_idx]) - 1) * 100
                    else:
                        rolling_return = (np.prod(1 + self.processed_data['benchmark_returns']
                                        [:date_idx+1]) - 1) * 100
                    
                    sel.annotation.set_text(
                        f"Benchmark\nDate: {date_str}\n"
                        f"Daily Return: {daily_return:.2f}%\n"
                        f"Cumulative Return: {cum_return:.2f}%\n"
                        f"20D Return: {rolling_return:.2f}%"
                    )
                
                else:  # Excess returns
                    daily_excess = (self.processed_data['portfolio_returns'][date_idx] - 
                                self.processed_data['benchmark_returns'][date_idx]) * 100
                    
                    # Calculate rolling excess return metrics
                    window = 20
                    if date_idx >= window:
                        rolling_excess = np.sum(self.processed_data['portfolio_returns'][date_idx-window:date_idx] - 
                                            self.processed_data['benchmark_returns'][date_idx-window:date_idx]) * 100
                        rolling_tracking_error = np.std(self.processed_data['portfolio_returns'][date_idx-window:date_idx] - 
                                                    self.processed_data['benchmark_returns'][date_idx-window:date_idx]) * np.sqrt(252) * 100
                    else:
                        rolling_excess = np.sum(self.processed_data['portfolio_returns'][:date_idx+1] - 
                                            self.processed_data['benchmark_returns'][:date_idx+1]) * 100
                        rolling_tracking_error = np.std(self.processed_data['portfolio_returns'][:date_idx+1] - 
                                                    self.processed_data['benchmark_returns'][:date_idx+1]) * np.sqrt(252) * 100
                    
                    sel.annotation.set_text(
                        f"Excess Returns\nDate: {date_str}\n"
                        f"Daily Excess: {daily_excess:.2f}%\n"
                        f"Cumulative Excess: {y*100:.2f}%\n"
                        f"20D Excess: {rolling_excess:.2f}%\n"
                        f"20D Track. Error: {rolling_tracking_error:.2f}%"
                    )
                
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

        return ax

    def plot_performance_summary(self, interactive=True):
        """Create comprehensive performance summary plot with interactive features"""
        if not hasattr(self, 'processed_data'):
            self.calculate_performance_metrics()

        # Create figure with appropriate dimensions
        fig = plt.figure(figsize=(16, 24))
        
        # Create grid with better proportions
        gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 1, 1, 1])

        # Performance Summary Table - Now wider and two rows
        ax_summary = fig.add_subplot(gs[0])
        ax_summary.axis('off')
        
        # Organize metrics in a single row
        metric_names = [
            'Total\nReturn (%)', 'Annualized\nReturn (%)', 'Sharpe\nRatio', 
            'Information\nRatio', 'Max\nDrawdown (%)', 'Benchmark Ann.\nReturn (%)',
            'Std of Excess\nReturn (%)', 'Avg Daily\nTurnover (%)', 'Number of\nTrades'
        ]
        
        metric_values = [
            f"{self.metrics['Total Return (%)']:.2f}",
            f"{self.metrics['Annualized Return (%)']:.2f}",
            f"{self.metrics['Sharpe Ratio']:.2f}",
            f"{self.metrics['Information Ratio']:.2f}",
            f"{self.metrics['Maximum Drawdown (%)']:.2f}",
            f"{self.metrics['Benchmark Annualized Return (%)']:.2f}",
            f"{self.metrics['Std of Excess Return (%)']:.2f}",
            f"{self.metrics['Avg Daily Turnover (%)']:.2f}",
            f"{self.metrics['Number of Trades']:.0f}"
        ]
        
        # Create table with names on top and values below
        table = ax_summary.table(
            cellText=[metric_values],
            cellLoc='center',
            colLabels=metric_names,
            bbox=[0.05, 0.2, 0.9, 0.6]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Style cells
        for (row, col), cell in table._cells.items():
            cell.set_edgecolor('white')
            if row == 0:  # Header row (metric names)
                cell.set_facecolor('#E6F3FF')
                cell.set_height(0.15)  # Make header cells taller for wrapped text
            else:  # Value row
                cell.set_facecolor('#F5F5F5')

        # Cumulative Returns Plot
        ax1 = fig.add_subplot(gs[1])
        self.plot_cumulative_returns(ax=ax1, interactive=interactive)

        # Drawdown Plot
        ax2 = fig.add_subplot(gs[2])
        self.plot_drawdown(ax2, interactive=True)

        # Turnover Plot
        ax3 = fig.add_subplot(gs[3])
        self.plot_turnover(ax3, interactive=True)

        plt.tight_layout()
        return fig

    def plot_drawdown(self, ax=None, interactive=True):
        """
        Plot drawdown analysis with interactive features, including excess return drawdowns
        and maximum drawdown markers for all series.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate excess return drawdown
        excess_returns = np.array(self.processed_data['portfolio_returns']) - np.array(self.processed_data['benchmark_returns'])
        cum_excess = np.cumprod(1 + excess_returns)
        excess_peak = np.maximum.accumulate(cum_excess)
        excess_drawdown = (cum_excess - excess_peak) / excess_peak * 100

        # Calculate maximum drawdowns
        portfolio_mdd = np.min(self.processed_data['portfolio_drawdown'])
        benchmark_mdd = np.min(self.processed_data['benchmark_drawdown'])
        excess_mdd = np.min(excess_drawdown)

        # Find maximum drawdown periods
        portfolio_mdd_idx = np.argmin(self.processed_data['portfolio_drawdown'])
        benchmark_mdd_idx = np.argmin(self.processed_data['benchmark_drawdown'])
        excess_mdd_idx = np.argmin(excess_drawdown)

        # Plot lines
        portfolio_line = ax.plot(
            self.processed_data['dates'],
            self.processed_data['portfolio_drawdown'],
            label=f'Portfolio (Max DD: {portfolio_mdd:.1f}%)',
            color='blue',
            linewidth=2
        )[0]

        benchmark_line = ax.plot(
            self.processed_data['dates'],
            self.processed_data['benchmark_drawdown'],
            label=f'Benchmark (Max DD: {benchmark_mdd:.1f}%)',
            color='red',
            linestyle='--',
            linewidth=1.5
        )[0]

        excess_line = ax.plot(
            self.processed_data['dates'],
            excess_drawdown,
            label=f'Excess (Max DD: {excess_mdd:.1f}%)',
            color='green',
            linestyle=':',
            linewidth=1.5
        )[0]

        # Mark maximum drawdown points
        ax.scatter(self.processed_data['dates'][portfolio_mdd_idx], portfolio_mdd,
                color='blue', s=100, zorder=5, alpha=0.6)
        ax.scatter(self.processed_data['dates'][benchmark_mdd_idx], benchmark_mdd,
                color='red', s=100, zorder=5, alpha=0.6)
        ax.scatter(self.processed_data['dates'][excess_mdd_idx], excess_mdd,
                color='green', s=100, zorder=5, alpha=0.6)

        ax.set_title('Drawdown Analysis', fontsize=12, pad=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        if interactive:
            cursor = mplcursors.cursor([portfolio_line, benchmark_line, excess_line], hover=True)
            
            @cursor.connect("add")
            def on_hover(sel):
                x, y = sel.target
                date_idx = np.abs(matplotlib.dates.date2num(self.processed_data['dates']) - x).argmin()
                date_str = self.processed_data['dates'][date_idx].strftime('%Y-%m-%d')
                
                if sel.artist == portfolio_line:
                    recovery_days = 0
                    peak_value = 0
                    
                    # Calculate drawdown length by counting consecutive negative days up to current point
                    drawdown_length = 0
                    for i in range(date_idx, -1, -1):
                        if self.processed_data['portfolio_drawdown'][i] >= 0:
                            break
                        drawdown_length += 1
                    
                    # Calculate days to recovery
                    for i in range(date_idx, len(self.processed_data['portfolio_drawdown'])):
                        if self.processed_data['portfolio_drawdown'][i] >= 0:
                            break
                        recovery_days += 1
                    
                    sel.annotation.set_text(
                        f"Portfolio Drawdown\nDate: {date_str}\n"
                        f"Drawdown: {y:.2f}%\n"
                        f"Length: {drawdown_length} days\n"
                        f"Days to Recovery: {recovery_days if y < 0 else 0}"
                    )
                elif sel.artist == benchmark_line:
                    drawdown_length = sum(1 for i in range(date_idx, -1, -1) 
                                    if self.processed_data['benchmark_drawdown'][i] < 0)
                    sel.annotation.set_text(
                        f"Benchmark Drawdown\nDate: {date_str}\n"
                        f"Drawdown: {y:.2f}%\n"
                        f"Length: {drawdown_length} days"
                    )
                else:  # Excess drawdown
                    drawdown_length = sum(1 for i in range(date_idx, -1, -1) 
                                    if excess_drawdown[i] < 0)
                    sel.annotation.set_text(
                        f"Excess Drawdown\nDate: {date_str}\n"
                        f"Drawdown: {y:.2f}%\n"
                        f"Length: {drawdown_length} days"
                    )
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

        return ax

    def plot_turnover(self, ax=None, interactive=True):
        """Plot turnover analysis with interactive features using scatter plot"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Create scatter plot for turnover
        turnover_scatter = ax.scatter(
            self.processed_data['dates'],
            self.processed_data['turnover'],
            color='blue',
            alpha=0.6,
            s=30,  # Size of points
            label=f'Daily Turnover (Avg: {self.metrics["Avg Daily Turnover (%)"]:.1f}%)'
        )

        avg_turnover = self.metrics["Avg Daily Turnover (%)"]
        ax.axhline(
            y=avg_turnover,
            color='red',
            linestyle='--',
            alpha=0.5,
            label=f'Average ({avg_turnover:.1f}%)'
        )

        ax.set_title('Portfolio Turnover', fontsize=12, pad=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Turnover (%)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        if interactive:
            cursor = mplcursors.cursor(turnover_scatter, hover=True)
            
            @cursor.connect("add")
            def on_hover(sel):
                x, y = sel.target
                date_idx = np.abs(matplotlib.dates.date2num(self.processed_data['dates']) - x).argmin()
                date_str = self.processed_data['dates'][date_idx].strftime('%Y-%m-%d')

                # Calculate rolling average turnover
                window = 20  # 20-day rolling average
                if date_idx >= window:
                    rolling_avg = np.mean(self.processed_data['turnover'][date_idx-window:date_idx])
                else:
                    rolling_avg = np.mean(self.processed_data['turnover'][:date_idx+1])

                sel.annotation.set_text(
                    f"Date: {date_str}\n"
                    f"Turnover: {y:.2f}%\n"
                    f"vs. Average: {(y - avg_turnover):.2f}%\n"
                    f"20D Avg: {rolling_avg:.2f}%"
                )
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

        return ax

    def save_performance_summary_pdfs(self, output_dir='performance_plots', dpi=600):
        """
        Save each row of the performance summary plot as a separate high-resolution PDF.
        
        Parameters:
        -----------
        output_dir : str
            Directory where PDF files will be saved
        dpi : int
            Resolution of the output PDFs (default: 600 for high quality)
        """
        import os
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Ensure metrics are calculated
        if not hasattr(self, 'processed_data'):
            self.calculate_performance_metrics()

        # 1. Save Metrics Table
        fig = plt.figure(figsize=(16, 4))
        ax_summary = fig.add_subplot(111)
        ax_summary.axis('off')
        
        metric_names = [
            'Total\nReturn (%)', 'Annualized\nReturn (%)', 'Sharpe\nRatio', 
            'Information\nRatio', 'Max\nDrawdown (%)', 'Benchmark Ann.\nReturn (%)',
            'Std of Excess\nReturn (%)', 'Avg Daily\nTurnover (%)', 'Number of\nTrades'
        ]
        
        metric_values = [
            f"{self.metrics['Total Return (%)']:.2f}",
            f"{self.metrics['Annualized Return (%)']:.2f}",
            f"{self.metrics['Sharpe Ratio']:.2f}",
            f"{self.metrics['Information Ratio']:.2f}",
            f"{self.metrics['Maximum Drawdown (%)']:.2f}",
            f"{self.metrics['Benchmark Annualized Return (%)']:.2f}",
            f"{self.metrics['Std of Excess Return (%)']:.2f}",
            f"{self.metrics['Avg Daily Turnover (%)']:.2f}",
            f"{self.metrics['Number of Trades']:.0f}"
        ]
        
        table = ax_summary.table(
            cellText=[metric_values],
            cellLoc='center',
            colLabels=metric_names,
            bbox=[0.05, 0.2, 0.9, 0.6]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        for (row, col), cell in table._cells.items():
            cell.set_edgecolor('white')
            if row == 0:
                cell.set_facecolor('#E6F3FF')
                cell.set_height(0.15)
            else:
                cell.set_facecolor('#F5F5F5')
        
        plt.savefig(os.path.join(output_dir, '1_metrics_table.pdf'), 
                    dpi=dpi, bbox_inches='tight')
        plt.close()

        # 2. Save Cumulative Returns Plot
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        self.plot_cumulative_returns(ax=ax, interactive=False)
        plt.savefig(os.path.join(output_dir, '2_cumulative_returns.pdf'), 
                    dpi=dpi, bbox_inches='tight')
        plt.close()

        # 3. Save Drawdown Plot
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        self.plot_drawdown(ax=ax, interactive=False)
        plt.savefig(os.path.join(output_dir, '3_drawdown_analysis.pdf'), 
                    dpi=dpi, bbox_inches='tight')
        plt.close()

        # 4. Save Turnover Plot
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        self.plot_turnover(ax=ax, interactive=False)
        plt.savefig(os.path.join(output_dir, '4_turnover_analysis.pdf'), 
                    dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"PDF files have been saved to {output_dir}/:")
        print("├── 1_metrics_table.pdf")
        print("├── 2_cumulative_returns.pdf")
        print("├── 3_drawdown_analysis.pdf")
        print("└── 4_turnover_analysis.pdf")

    
