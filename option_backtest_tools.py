
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import option_tools as opt
import duckdb
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')



class OptionCommissionModel:
    """Commission model for options trading"""
    
    def __init__(self, per_contract_fee=0.65, min_commission=1.0, base_fee=0.0):
        """
        Initialize options commission model
        
        Parameters:
        -----------
        per_contract_fee : float
            Fee per option contract
        min_commission : float
            Minimum commission per trade
        base_fee : float
            Base fee per trade (added before per-contract fees)
        """
        self.per_contract_fee = per_contract_fee
        self.min_commission = min_commission
        self.base_fee = base_fee
        
    def calculate_commission(self, contracts, price_per_share=None):
        """
        Calculate commission for an option trade
        
        Parameters:
        -----------
        contracts : int
            Number of option contracts traded
        price_per_share : float, optional
            Price per share (not used in this model, but included for compatibility)
            
        Returns:
        --------
        float : Commission amount
        """
        # Calculate the commission based on number of contracts
        commission = self.base_fee + (abs(contracts) * self.per_contract_fee)
        
        # Apply minimum commission
        commission = max(commission, self.min_commission)
        
        return commission
    

class OptionTrading:
    """
    Class for backtesting option trading strategies.
    Handles option position management, expiry processing, and P&L tracking.
    """
    def __init__(self, cash, commission_model, db_path, option_data_dic=None,is_print=False, txt_file=None):
        """
        Initialize options trading system
        
        Parameters:
        -----------
        cash : float
            Initial cash
        commission_model : CommissionModel
            Commission model for option trades
        db_path : str
            Path to the database file for fetching option data
        is_print : bool
            Whether to print logs to console
        txt_file : file object
            File to write logs to
        """
        self.cash = cash
        self.initial_cash = cash
        self.commission_model = commission_model
        self.db_path = db_path
        self.option_data_dic = option_data_dic 
        # Position tracking - {optionid: [cost_basis, contracts, current_price, entry_date, expiry_date, option_type, strike, is_long, contract_size, last_cfadj]}
        # Added last_cfadj to track corporate actions
        self.option_positions = {}
        
        # Performance tracking
        self.daily_value_list = [cash]
        self.daily_returns = [0.0]
        self.daily_cash = [cash]
        self.trades_history = []
        self.expired_options = []
                # Component P&L tracking

        self.daily_component_pnl = {
            'long_calls': [0.0],
            'long_puts': [0.0],
            'short_calls': [0.0],
            'short_puts': [0.0],
            'total': [0.0]
        }
        

        # Previous day's positions value by component
        self.previous_component_value = {
            'long_calls': 0.0,
            'long_puts': 0.0,
            'short_calls': 0.0,
            'short_puts': 0.0
        }

        # Corporate action tracking
        self.corporate_action_history = []
        
        # Current targets tracking
        self.current_targets = {}
        
        # Add logging parameters
        self.is_print = is_print
        self.txt_file = txt_file
        
    def logger(self, txt_str):
        """Log message to file and optionally print"""
        if self.txt_file is not None:
            print(txt_str, file=self.txt_file)
        if self.is_print:
            print(txt_str)
    
    def open_long_position(self, date, option_data, contracts=None, weight=None, metadata=None):
        """
        Open a long position for an option contract
        
        Parameters:
        -----------
        date : str
            Trade date in 'YYYY-MM-DD' format
        option_data : Series or dict
            Option data
        contracts : int, optional
            Number of contracts to long. If None, calculated from weight.
        weight : float, optional
            Portfolio weight to allocate. Ignored if contracts is provided.
        metadata : dict, optional
            Additional metadata to store with the position
            
        Returns:
        --------
        bool : Whether the trade was executed successfully
        """
        optionid = option_data['optionid']
        
        # Get appropriate price - includes contract_size
        price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
            option_data, is_opening=True, is_long=True
        )
        
        if price_per_share <= 0:
            self.logger(f"Invalid price ({price_per_share}) for option {optionid}")
            return False
        
        # Calculate contracts if not provided
        if contracts is None and weight is not None:
            portfolio_value = self.get_portfolio_value()
            contracts = opt.calculate_contracts(
                weight, price_per_share, portfolio_value, contract_size
            )
        
        if contracts <= 0:
            self.logger(f"Invalid contract count ({contracts}) for option {optionid}")
            return False
        
        # Calculate total cost
        total_cost = contracts * price_per_contract
        
        # Calculate commission - using actual contract size
        commission = self.commission_model.calculate_commission(
            contracts , price_per_share
        )
        total_cost += commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            self.logger(f"Insufficient cash (${self.cash:.2f}) for option trade (${total_cost:.2f})")
            return False
        
        # Get the current cfadj value for future adjustment tracking
        current_cfadj = option_data.get('cfadj', 1.0)
        if pd.isna(current_cfadj) or current_cfadj <= 0:
            current_cfadj = 1.0
        
        # Execute trade
        if optionid in self.option_positions:
            # Update existing position
            old_basis = self.option_positions[optionid][0]
            old_contracts = self.option_positions[optionid][1]
            
            # Check if position is long
            if not self.option_positions[optionid][7]:  # is_long flag
                self.logger(f"Cannot add long position to existing short position for option {optionid}")
                return False
                
            new_contracts = old_contracts + contracts
            new_basis = ((old_basis * old_contracts) + (price_per_contract * contracts)) / new_contracts
            self.option_positions[optionid][0] = new_basis
            self.option_positions[optionid][1] = new_contracts
            self.option_positions[optionid][2] = price_per_contract  # Update current price
            
            # Update metadata if provided and position has metadata field
            if metadata is not None and len(self.option_positions[optionid]) > 10:
                self.option_positions[optionid][10].update(metadata)
        else:
            # Create position list
            position = [
                price_per_contract,  # cost_basis
                contracts,           # contracts
                price_per_contract,  # current_price
                date,                # entry_date
                option_data['exdate'], # expiry_date
                option_data['cp_flag'], # option_type
                option_data['strike_price'], # strike
                True,                # is_long flag
                contract_size,       # contract_size
                current_cfadj        # last_cfadj
            ]
            
            # Add metadata if provided
            if metadata is not None:
                position.append(metadata)
            
            # Create new position
            self.option_positions[optionid] = position
        
        # Update cash
        self.cash -= total_cost
        
        # Record trade
        self.trades_history.append({
            'date': date,
            'optionid': optionid,
            'secid': option_data.get('secid'),
            'permno': option_data.get('permno'),
            'action': 'OPEN_LONG',
            'contracts': contracts,
            'contract_size': contract_size,
            'price_per_share': price_per_share,
            'price_per_contract': price_per_contract,
            'commission': commission,
            'option_type': option_data['cp_flag'],
            'strike': option_data['strike_price'],
            'expiry_date': option_data['exdate'],
            'total_cost': total_cost,
            'cfadj': current_cfadj
        })
        
        self.logger(f"Opened long position of {contracts} contracts for option {optionid} at ${price_per_contract:.2f} per contract")
        return True
    
    def close_long_position(self, date, option_data, contracts=None):
        """
        Close a long position for an option contract
        
        Parameters:
        -----------
        date : str
            Trade date in 'YYYY-MM-DD' format
        option_data : Series or dict
            Option data
        contracts : int, optional
            Number of contracts to close. If None, closes all contracts.
            
        Returns:
        --------
        bool : Whether the trade was executed successfully
        """
        optionid = option_data['optionid']
        
        if optionid not in self.option_positions:
            self.logger(f"No position found for option {optionid}")
            return False
        
        # Check if position is long
        if not self.option_positions[optionid][7]:  # is_long flag
            self.logger(f"Cannot close long position for option {optionid} - position is short")
            return False
            
        current_contracts = self.option_positions[optionid][1]
        contract_size = self.option_positions[optionid][8]
        
        # If contracts not specified, close all
        if contracts is None:
            contracts = current_contracts
        
        if contracts > current_contracts:
            contracts = current_contracts  # Adjust to available contracts
        
        # Get appropriate price
        price_per_share, price_per_contract, _ = opt.calculate_option_price(
            option_data, is_opening=False, is_long=True
        )
        
        if price_per_share <= 0:
            self.logger(f"Invalid price ({price_per_share}) for option {optionid}")
            return False
        
        # Calculate proceeds
        total_proceeds = contracts * price_per_contract
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(
            contracts , price_per_share
        )
        net_proceeds = total_proceeds - commission
        
        # Calculate P&L
        cost_basis = self.option_positions[optionid][0]
        pl_per_contract = price_per_contract - cost_basis
        total_pl = pl_per_contract * contracts
        
        # Update position
        if contracts == current_contracts:
            # Close position
            del self.option_positions[optionid]
        else:
            # Reduce position
            self.option_positions[optionid][1] -= contracts
        
        # Update cash
        self.cash += net_proceeds
        
        # Record trade
        self.trades_history.append({
            'date': date,
            'optionid': optionid,
            'secid': option_data.get('secid'),
            'permno': option_data.get('permno'),
            'action': 'CLOSE_LONG',
            'contracts': contracts,
            'contract_size': contract_size,
            'price_per_share': price_per_share,
            'price_per_contract': price_per_contract,
            'commission': commission,
            'cost_basis': cost_basis,
            'pl_per_contract': pl_per_contract,
            'total_pl': total_pl,
            'option_type': option_data['cp_flag'],
            'strike': option_data['strike_price'],
            'expiry_date': option_data['exdate'],
            'net_proceeds': net_proceeds,
            'cfadj': option_data.get('cfadj', 1.0)
        })
        
        self.logger(f"Closed long position of {contracts} contracts for option {optionid} at ${price_per_contract:.2f} per contract (P&L: ${total_pl:.2f})")
        return True
    
    def open_short_position(self, date, option_data, contracts=None, weight=None, metadata=None):
        """
        Open a short position for an option contract
        
        Parameters:
        -----------
        date : str
            Trade date in 'YYYY-MM-DD' format
        option_data : Series or dict
            Option data
        contracts : int, optional
            Number of contracts to short. If None, calculated from weight.
        weight : float, optional
            Portfolio weight to allocate. Ignored if contracts is provided.
        metadata : dict, optional
            Additional metadata to store with the position
            
        Returns:
        --------
        bool : Whether the trade was executed successfully
        """
        optionid = option_data['optionid']
        
        # Get appropriate price
        price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
            option_data, is_opening=True, is_long=False
        )
        
        if price_per_share <= 0:
            self.logger(f"Invalid price ({price_per_share}) for option {optionid}")
            return False
        
        # Calculate contracts if not provided
        if contracts is None and weight is not None:
            portfolio_value = self.get_portfolio_value()
            contracts = opt.calculate_contracts(
                weight, price_per_share, portfolio_value, contract_size
            )
            
        if contracts <= 0:
            self.logger(f"Invalid contract count ({contracts}) for option {optionid}")
            return False
        
        # Calculate proceeds per contract
        total_proceeds = contracts * price_per_contract
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(
            contracts , price_per_share
        )
        net_proceeds = total_proceeds - commission
        
        # Get the current cfadj value for future adjustment tracking
        current_cfadj = option_data.get('cfadj', 1.0)
        if pd.isna(current_cfadj) or current_cfadj <= 0:
            current_cfadj = 1.0
            
        # For short positions, store contract count as positive but flag as short
        if optionid in self.option_positions:
            # Update existing position
            old_basis = self.option_positions[optionid][0]
            old_contracts = self.option_positions[optionid][1]
            
            # Check if position is short
            if self.option_positions[optionid][7]:  # is_long flag
                self.logger(f"Cannot add short position to existing long position for option {optionid}")
                return False
                
            new_contracts = old_contracts + contracts
            new_basis = ((old_basis * old_contracts) + (price_per_contract * contracts)) / new_contracts
            self.option_positions[optionid][0] = new_basis
            self.option_positions[optionid][1] = new_contracts
            self.option_positions[optionid][2] = price_per_contract  # Update current price
            
            # Update metadata if provided and position has metadata field
            if metadata is not None and len(self.option_positions[optionid]) > 10:
                self.option_positions[optionid][10].update(metadata)
        else:
            # Create position list
            position = [
                price_per_contract,  # opening_price
                contracts,           # contracts
                price_per_contract,  # current_price
                date,                # entry_date
                option_data['exdate'], # expiry_date
                option_data['cp_flag'], # option_type
                option_data['strike_price'], # strike
                False,               # is_long flag = False for short positions
                contract_size,       # contract_size
                current_cfadj        # last_cfadj
            ]
            
            # Add metadata if provided
            if metadata is not None:
                position.append(metadata)
            
            # Create new short position
            self.option_positions[optionid] = position
        
        # Update cash (shorting adds cash)
        self.cash += net_proceeds
        
        # Record trade
        self.trades_history.append({
            'date': date,
            'optionid': optionid,
            'secid': option_data.get('secid'),
            'permno': option_data.get('permno'),
            'action': 'OPEN_SHORT',
            'contracts': contracts,
            'contract_size': contract_size,
            'price_per_share': price_per_share,
            'price_per_contract': price_per_contract,
            'commission': commission,
            'option_type': option_data['cp_flag'],
            'strike': option_data['strike_price'],
            'expiry_date': option_data['exdate'],
            'net_proceeds': net_proceeds,
            'cfadj': current_cfadj
        })
        
        self.logger(f"Opened short position of {contracts} contracts for option {optionid} at ${price_per_contract:.2f} per contract")
        return True
    '''    
    def close_short_position(self, date, option_data, contracts=None):
        """
        Close a short position for an option contract
        
        Parameters:
        -----------
        date : str
            Trade date in 'YYYY-MM-DD' format
        option_data : Series or dict
            Option data
        contracts : int, optional
            Number of contracts to close (positive number). If None, closes all.
            
        Returns:
        --------
        bool : Whether the trade was executed successfully
        """
        optionid = option_data['optionid']
        
        if optionid not in self.option_positions:
            self.logger(f"No position found for option {optionid}")
            return False
        
        # Verify this is a short position
        if self.option_positions[optionid][7]:  # is_long flag
            self.logger(f"Position for option {optionid} is not a short position")
            return False
            
        current_contracts = self.option_positions[optionid][1]
        contract_size = self.option_positions[optionid][8]
            
        if contracts is None:
            contracts = current_contracts
        elif contracts > current_contracts:
            contracts = current_contracts  # Adjust to available contracts
        
        # Get appropriate price for closing shorts (buying back)
        price_per_share, price_per_contract, _ = opt.calculate_option_price(
            option_data, is_opening=False, is_long=False
        )
        
        if price_per_share <= 0:
            self.logger(f"Invalid price ({price_per_share}) for option {optionid}")
            return False
        
        # Calculate cost to close
        total_cost = contracts * price_per_contract
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(
            contracts * contract_size, price_per_share
        )
        total_cost += commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            self.logger(f"Insufficient cash (${self.cash:.2f}) to close short position (${total_cost:.2f})")
            return False
        
        # Calculate P&L
        opening_price = self.option_positions[optionid][0]
        pl_per_contract = opening_price - price_per_contract  # For shorts, profit is entry price minus exit price
        total_pl = pl_per_contract * contracts
        
        # Update position
        if contracts == current_contracts:
            # Close position
            del self.option_positions[optionid]
        else:
            # Reduce position
            self.option_positions[optionid][1] -= contracts
        
        # Update cash
        self.cash -= total_cost
        
        # Record trade
        self.trades_history.append({
            'date': date,
            'optionid': optionid,
            'secid': option_data.get('secid'),
            'permno': option_data.get('permno'),
            'action': 'CLOSE_SHORT',
            'contracts': contracts,
            'contract_size': contract_size,
            'price_per_share': price_per_share,
            'price_per_contract': price_per_contract,
            'commission': commission,
            'opening_price': opening_price,
            'pl_per_contract': pl_per_contract,
            'total_pl': total_pl,
            'option_type': option_data['cp_flag'],
            'strike': option_data['strike_price'],
            'expiry_date': option_data['exdate'],
            'total_cost': total_cost,
            'cfadj': option_data.get('cfadj', 1.0)
        })
        
        self.logger(f"Closed short position of {contracts} contracts for option {optionid} at ${price_per_contract:.2f} per contract (P&L: ${total_pl:.2f})")
        return True
    '''
    def handle_option_expiry(self, date):
        """
        Handle option expiry for all positions
        
        Parameters:
        -----------
        date : str
            Current date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        None
        """

        
        for optionid in list(self.option_positions.keys()):
            position = self.option_positions[optionid]
            expiry_date = position[4]
            
            # Check if option is expired
            if expiry_date == date:
                option_type = position[5]
                strike = position[6]
                contracts = position[1]
                cost_basis = position[0]
                is_long = position[7]
                contract_size = position[8]
                
                # Get option data for the final day - use preloaded data if available
                if self.option_data_dic is not None:
                    option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
                else:
                    option_data = opt.load_option_by_id(date, optionid, self.db_path)                
                # Get underlying stock price
                underlying_price = None
                permno = None
                
                if option_data is not None:
                    permno = option_data.get('permno')
                    
                    # Try to get stock price from database with a dedicated connection
                    if permno is not None:
                        try:
                            with duckdb.connect(self.db_path, read_only=True) as db_conn:
                                stock_query = f"""
                                SELECT dlyprc, dlycloseprc
                                FROM dsf_v2
                                WHERE permno = {permno}
                                AND dlycaldt = DATE '{date}'
                                """
                                stock_result = db_conn.execute(stock_query).fetchdf()
                                if len(stock_result) > 0:
                                    # Use closing price
                                    underlying_price = stock_result['dlycloseprc'].iloc[0]
                                    # If closing price not available, use regular price
                                    if pd.isna(underlying_price):
                                        underlying_price = stock_result['dlyprc'].iloc[0]
                        except Exception as e:
                            self.logger(f"Error getting stock price for expiration: {str(e)}")
                
                # If no stock price found in database, try to infer from option data
                if underlying_price is None and option_data is not None:
                    # Infer from final option values
                    if option_type == 'C' and 'delta' in option_data and not pd.isna(option_data['delta']):
                        delta = option_data['delta']
                        # For a call that's ATM (delta ~0.5), stock price ~= strike
                        # As delta approaches 1, stock price > strike; as delta approaches 0, stock price < strike
                        underlying_price = strike + (delta - 0.5) * 2 * strike * 0.1
                    elif option_type == 'P' and 'delta' in option_data and not pd.isna(option_data['delta']):
                        delta = option_data['delta']
                        # For a put that's ATM (delta ~-0.5), stock price ~= strike
                        # As delta approaches 0, stock price > strike; as delta approaches -1, stock price < strike
                        underlying_price = strike + (delta + 0.5) * 2 * strike * 0.1
                
                # If still no price, use direct option pricing
                if underlying_price is None and option_data is not None:
                    # Calculate intrinsic value at expiration
                    if 'midprice' in option_data and not pd.isna(option_data['midprice']):
                        intrinsic_value = option_data['midprice']
                    elif 'bidprice' in option_data and 'askprice' in option_data:
                        # Use midpoint
                        intrinsic_value = (option_data['bidprice'] + option_data['askprice']) / 2
                    else:
                        intrinsic_value = 0
                        
                    # For puts, calculate stock price
                    if option_type == 'P' and intrinsic_value > 0:
                        underlying_price = strike - intrinsic_value
                    # For calls, calculate stock price
                    elif option_type == 'C' and intrinsic_value > 0:
                        underlying_price = strike + intrinsic_value
                    else:
                        # If option has no value, stock price is approximately strike
                        underlying_price = strike
                
                # If still no price, default to strike price (ATM)
                if underlying_price is None:
                    self.logger(f"Warning: No price data for option {optionid} expiration. Using strike price as approximation.")
                    underlying_price = strike
                
                # Determine if this is a long or short position
                position_type = "LONG" if is_long else "SHORT"
                
                # Calculate expiry value
                if option_type == 'C':  # Call option
                    intrinsic_value = max(0, underlying_price - strike)
                elif option_type == 'P':  # Put option
                    intrinsic_value = max(0, strike - underlying_price)
                else:
                    intrinsic_value = 0
                
                # Calculate per contract and total values
                expiry_value = intrinsic_value * contract_size
                total_value = expiry_value * contracts
                
                # Calculate P&L
                if is_long:
                    # For long positions: expiry value - cost basis
                    total_cost = cost_basis * contracts
                    total_pl = total_value - total_cost
                    
                    # Update cash
                    self.cash += total_value
                else:
                    # For short positions: collected premium - expiry value
                    collected_premium = cost_basis * contracts
                    total_pl = collected_premium - total_value
                    
                    # Update cash (pay out obligation)
                    self.cash -= total_value
                
                # Record expiry
                expiry_record = {
                    'date': date,
                    'optionid': optionid,
                    'permno': permno,
                    'position_type': position_type,
                    'option_type': option_type,
                    'strike': strike,
                    'contracts': contracts,
                    'contract_size': contract_size,
                    'cost_basis': cost_basis,
                    'underlying_price': underlying_price,
                    'intrinsic_value': intrinsic_value,
                    'expiry_value': expiry_value,
                    'total_value': total_value,
                    'total_pl': total_pl
                }
                
                self.expired_options.append(expiry_record)
                
                # Remove from positions
                del self.option_positions[optionid]
                
                self.logger(f"{position_type} option {optionid} expired: {contracts} {option_type} contracts at strike ${strike:.2f}")
                self.logger(f"  Underlying price: ${underlying_price:.2f}, Expiry value: ${expiry_value:.2f} per contract")
                self.logger(f"  Total P&L: ${total_pl:.2f}")
    
    def handle_corporate_actions(self, date):
        """
        Check for and handle corporate actions for all option positions
        
        Parameters:
        -----------
        date : str
            Current date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        None
        """
        self.logger(f"\nChecking for corporate actions on {date}...")
        
        if not self.option_positions:
            return
            
        # Load current data for all options in our positions
        optionids = list(self.option_positions.keys())
        if self.option_data_dic is not None:
            current_data = opt.load_option_batch(date, optionids, self.db_path, self.option_data_dic)
        else:
            current_data = opt.load_option_batch(date, optionids, self.db_path)
        
        if len(current_data) == 0:
            return
            
        # Create lookup dictionary
        option_lookup = {row['optionid']: row for _, row in current_data.iterrows()}
        
        # Process each position for corporate actions
        for optionid in list(self.option_positions.keys()):
            if optionid not in option_lookup:
                continue
                
            position = self.option_positions[optionid]
            last_cfadj = position[9]  # Last known adjustment factor
            current_data = option_lookup[optionid]
            current_cfadj = current_data.get('cfadj', 1.0)
            
            # Check for NaN or invalid adjustment factors
            if pd.isna(current_cfadj) or current_cfadj <= 0:
                current_cfadj = 1.0
                
            # If adjustment factor has changed, adjust the position
            if current_cfadj != last_cfadj and current_cfadj > 0:
                self.logger(f"Detected corporate action for option {optionid}: cfadj changed from {last_cfadj} to {current_cfadj}")
                
                # Calculate adjustment ratio
                adjustment_ratio = current_cfadj / last_cfadj
                
                # Adjust contract size
                old_contract_size = position[8]
                new_contract_size = old_contract_size * adjustment_ratio
                
                # Adjust strike price (inversely proportional to cfadj)
                old_strike = position[6]
                new_strike = old_strike / adjustment_ratio
                
                # Adjust cost basis (proportional to cfadj for cash value)
                old_cost_basis = position[0]
                new_cost_basis = old_cost_basis * adjustment_ratio
                
                # Adjust current price
                old_price = position[2]
                new_price = old_price * adjustment_ratio
                
                # Update position
                self.option_positions[optionid][0] = new_cost_basis  # cost_basis
                self.option_positions[optionid][2] = new_price  # current_price
                self.option_positions[optionid][6] = new_strike  # strike
                self.option_positions[optionid][8] = new_contract_size  # contract_size
                self.option_positions[optionid][9] = current_cfadj  # last_cfadj
                
                # Record adjustment
                self.corporate_action_history.append({
                    'date': date,
                    'optionid': optionid,
                    'old_cfadj': last_cfadj,
                    'new_cfadj': current_cfadj,
                    'old_contract_size': old_contract_size,
                    'new_contract_size': new_contract_size,
                    'old_strike': old_strike,
                    'new_strike': new_strike,
                    'old_cost_basis': old_cost_basis,
                    'new_cost_basis': new_cost_basis
                })
                
                self.logger(f"  Adjusted position:")
                self.logger(f"    Strike: ${old_strike:.2f} -> ${new_strike:.2f}")
                self.logger(f"    Contract Size: {old_contract_size:.2f} -> {new_contract_size:.2f}")
                self.logger(f"    Cost Basis: ${old_cost_basis:.2f} -> ${new_cost_basis:.2f}")
    
    def update_option_prices(self, date):
        """
        Update prices for all option positions
        
        Parameters:
        -----------
        date : str
            Current date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        None
        """
        if not self.option_positions:
            return
        
        # First check for corporate actions and adjust positions accordingly
        self.handle_corporate_actions(date)
        
        # Now batch load all active option IDs
        optionids = list(self.option_positions.keys())
        if self.option_data_dic is not None:
            options_data = opt.load_option_batch(date, optionids, self.db_path, self.option_data_dic)
        else:
            options_data = opt.load_option_batch(date, optionids, self.db_path)
        
        if len(options_data) > 0:
            # Create lookup dictionary for easier access
            options_lookup = {row['optionid']: row for _, row in options_data.iterrows()}
            
            # Update each position
            for optionid in optionids:
                if optionid in options_lookup:
                    option_data = options_lookup[optionid]
                    
                    # Calculate mid price
                    if 'midprice' in option_data and not pd.isna(option_data['midprice']):
                        mid_price = option_data['midprice']
                    else:
                        bid = option_data.get('bidprice', 0) if not pd.isna(option_data.get('bidprice', 0)) else 0
                        ask = option_data.get('askprice', 0) if not pd.isna(option_data.get('askprice', 0)) else 0
                        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                    
                    if mid_price > 0:
                        # Get contract size from position
                        contract_size = self.option_positions[optionid][8]
                        
                        # Update current price in position
                        contract_price = mid_price * contract_size
                        self.option_positions[optionid][2] = contract_price
                    else:
                        self.logger(f"Warning: No valid price found for option {optionid}")
                else:
                    self.logger(f"Warning: No data found for option {optionid}")
    
    def get_portfolio_value(self):
        """
        Calculate total portfolio value including cash and option positions
        
        Returns:
        --------
        float : Portfolio value
        """
        # Start with cash
        total_value = self.cash
        
        # Add value of all positions
        for optionid, position in self.option_positions.items():
            contracts = position[1]
            current_price = position[2]
            total_value += contracts * current_price
            
        return total_value
    

    def get_component_values(self):
        """
        Calculate value of positions by component type
        
        Returns:
        --------
        dict : Value of positions by component
        """
        # Initialize component values
        component_values = {
            'long_calls': 0.0,
            'long_puts': 0.0,
            'short_calls': 0.0,
            'short_puts': 0.0
        }
        
        # Calculate values by component
        for optionid, position in self.option_positions.items():
            contracts = position[1]
            current_price = position[2]
            option_type = position[5]
            is_long = position[7]
            
            position_value = contracts * current_price
            
            # Categorize by option type and direction
            if is_long and option_type == 'C':
                component_values['long_calls'] += position_value
            elif is_long and option_type == 'P':
                component_values['long_puts'] += position_value
            elif not is_long and option_type == 'C':
                component_values['short_calls'] += position_value
            elif not is_long and option_type == 'P':
                component_values['short_puts'] += position_value
        
        return component_values
        
    def update_component_pnl(self):
        """
        Update daily P&L for each component
        
        Returns:
        --------
        dict : Daily P&L by component
        """
        # Get current component values
        current_values = self.get_component_values()
        
        # Calculate daily P&L by component
        daily_pnl = {}
        for component, value in current_values.items():
            prev_value = self.previous_component_value.get(component, 0.0)
            daily_pnl[component] = value - prev_value
        
        # Add trade P&L from today's trades
        for trade in self.trades_history:
            if 'date' in trade and pd.to_datetime(trade['date']) == pd.to_datetime(datetime.now().strftime('%Y-%m-%d')):
                # For opening positions, P&L already reflected in position value
                # For closing positions, add realized P&L
                if trade['action'] == 'CLOSE_LONG' or trade['action'] == 'CLOSE_SHORT':
                    total_pl = trade.get('total_pl', 0.0)
                    option_type = trade.get('option_type', '')
                    
                    # Add to appropriate component
                    if trade['action'] == 'CLOSE_LONG' and option_type == 'C':
                        daily_pnl['long_calls'] += total_pl
                    elif trade['action'] == 'CLOSE_LONG' and option_type == 'P':
                        daily_pnl['long_puts'] += total_pl
                    elif trade['action'] == 'CLOSE_SHORT' and option_type == 'C':
                        daily_pnl['short_calls'] += total_pl
                    elif trade['action'] == 'CLOSE_SHORT' and option_type == 'P':
                        daily_pnl['short_puts'] += total_pl
        
        # Add to daily component P&L tracking
        for component, pnl in daily_pnl.items():
            self.daily_component_pnl[component].append(pnl)
        
        # Calculate total P&L
        total_pnl = sum(daily_pnl.values())
        self.daily_component_pnl['total'].append(total_pnl)
        
        # Update previous component values for next day
        self.previous_component_value = current_values
        
        return daily_pnl
    '''   
    def process_option_targets(self, date, targets):
        """
        Process option targets by calculating contracts and executing trades
        
        Parameters:
        -----------
        date : str
            Current date in 'YYYY-MM-DD' format
        targets : dict
            Dictionary of option targets from strategy
            
        Returns:
        --------
        dict : Result summary
        """
        if not targets:
            return {'trades': 0, 'success': 0}
        
        # First, check if this is a rebalance day
        rebalance = targets.get('rebalance', False)
        
        # Get current portfolio value for weight calculations
        portfolio_value = self.get_portfolio_value()
        
        # Track trades
        trade_count = 0
        success_count = 0
        long_value = 0.0
        short_value = 0.0
        
        # Check for positions to close based on target date
        current_date = pd.to_datetime(date)
        for optionid in list(self.option_positions.keys()):
            position = self.option_positions[optionid]
            is_long = position[7]  # Is long flag
            
            # Check if position has a target close date from metadata
            position_metadata = position[10] if len(position) > 10 else {}
            if isinstance(position_metadata, dict) and 'target_date' in position_metadata:
                target_date = pd.to_datetime(position_metadata['target_date'])
                
                # If target date reached, close the position
                if current_date >= target_date:
                    self.logger(f"Closing position for option {optionid} - reached target holding period")
                    
                    # Get current option data
                    if self.option_data_dic is not None:
                        option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
                    else:
                        option_data = opt.load_option_by_id(date, optionid, self.db_path)
                    
                    if option_data is not None:
                        trade_count += 1
                        
                        if is_long:
                            success = self.close_long_position(date, option_data)
                        else:
                            success = self.close_short_position(date, option_data)
                            
                        if success:
                            success_count += 1
                            contracts = position[1]
                            current_price = position[2]
                            if is_long:
                                long_value += contracts * current_price
                            else:
                                short_value += contracts * current_price
        
        # If rebalancing, close all remaining positions
        if rebalance:
            # Get all optionids from current positions
            optionids = list(self.option_positions.keys())
            
            for optionid in optionids:
                position = self.option_positions[optionid]
                is_long = position[7]
                contracts = position[1]
                current_price = position[2]
                
                if self.option_data_dic is not None:
                    option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
                else:
                    option_data = opt.load_option_by_id(date, optionid, self.db_path)
                
                if option_data is not None:
                    trade_count += 1
                    
                    if is_long:
                        success = self.close_long_position(date, option_data)
                        if success:
                            success_count += 1
                            long_value += contracts * current_price
                    else:
                        success = self.close_short_position(date, option_data)
                        if success:
                            success_count += 1
                            short_value += contracts * current_price
                else:
                    self.logger(f"Warning: Cannot close position for option {optionid} - no data available")
        
        # Process long calls
        for option in targets.get('long_calls', []):
            optionid = option.get('optionid')
            weight = option.get('weight', 0.01)  # Default 1% weight
            target_date = option.get('target_date')  # When to close this position
            
            if self.option_data_dic is not None:
                option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
            else:
                option_data = opt.load_option_by_id(date, optionid, self.db_path)
            
            if option_data is not None:
                trade_count += 1
                
                # Store metadata for future reference
                metadata = {
                    'target_date': target_date,
                    'strategy': option.get('strategy', 'long_call')
                }
                
                success = self.open_long_position(date, option_data, weight=weight, metadata=metadata)
                if success:
                    success_count += 1
                    
                    # Calculate and update contracts in target
                    price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
                        option_data, is_opening=True, is_long=True
                    )
                    contracts = opt.calculate_contracts(weight, price_per_share, portfolio_value, contract_size)
                    option['contracts'] = contracts
                    long_value += contracts * price_per_contract
        
        # Process long puts
        for option in targets.get('long_puts', []):
            optionid = option.get('optionid')
            weight = option.get('weight', 0.01)  # Default 1% weight
            target_date = option.get('target_date')  # When to close this position
            
            if self.option_data_dic is not None:
                option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
            else:
                option_data = opt.load_option_by_id(date, optionid, self.db_path)
            
            if option_data is not None:
                trade_count += 1
                
                # Store metadata for future reference
                metadata = {
                    'target_date': target_date,
                    'strategy': option.get('strategy', 'long_put')
                }
                
                success = self.open_long_position(date, option_data, weight=weight, metadata=metadata)
                if success:
                    success_count += 1
                    
                    # Calculate and update contracts in target
                    price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
                        option_data, is_opening=True, is_long=True
                    )
                    contracts = opt.calculate_contracts(weight, price_per_share, portfolio_value, contract_size)
                    option['contracts'] = contracts
                    long_value += contracts * price_per_contract
        
        # Process short calls
        for option in targets.get('short_calls', []):
            optionid = option.get('optionid')
            weight = option.get('weight', 0.01)  # Default 1% weight
            target_date = option.get('target_date')  # When to close this position
            
            if self.option_data_dic is not None:
                option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
            else:
                option_data = opt.load_option_by_id(date, optionid, self.db_path)
            
            if option_data is not None:
                trade_count += 1
                
                # Store metadata for future reference
                metadata = {
                    'target_date': target_date,
                    'strategy': option.get('strategy', 'short_call')
                }
                
                success = self.open_short_position(date, option_data, weight=weight, metadata=metadata)
                if success:
                    success_count += 1
                    
                    # Calculate and update contracts in target
                    price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
                        option_data, is_opening=True, is_long=False
                    )
                    contracts = opt.calculate_contracts(weight, price_per_share, portfolio_value, contract_size)
                    option['contracts'] = contracts
                    short_value += contracts * price_per_contract
        
        # Process short puts
        for option in targets.get('short_puts', []):
            optionid = option.get('optionid')
            weight = option.get('weight', 0.01)  # Default 1% weight
            target_date = option.get('target_date')  # When to close this position
            
            if self.option_data_dic is not None:
                option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
            else:
                option_data = opt.load_option_by_id(date, optionid, self.db_path)
            
            if option_data is not None:
                trade_count += 1
                
                # Store metadata for future reference
                metadata = {
                    'target_date': target_date,
                    'strategy': option.get('strategy', 'short_put')
                }
                
                success = self.open_short_position(date, option_data, weight=weight, metadata=metadata)
                if success:
                    success_count += 1
                    
                    # Calculate and update contracts in target
                    price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
                        option_data, is_opening=True, is_long=False
                    )
                    contracts = opt.calculate_contracts(weight, price_per_share, portfolio_value, contract_size)
                    option['contracts'] = contracts
                    short_value += contracts * price_per_contract
        
        # Calculate total trade value for turnover calculation
        total_value = long_value + short_value
        
        return {
            'trades': trade_count,
            'success': success_count,
            'rebalance': rebalance,
            'long_value': long_value,
            'short_value': short_value,
            'total_value': total_value
        }
    '''
    def open_long_position_by_notional(self, date, option_data, notional=None, contracts=None, metadata=None):
        """
        Open a long position using notional amount instead of weight
        
        Parameters:
        -----------
        date : str
            Trade date in 'YYYY-MM-DD' format
        option_data : Series or dict
            Option data
        notional : float, optional
            Notional amount to invest (e.g., $10,000). If None, uses contracts.
        contracts : int, optional
            Number of contracts to long. Ignored if notional is provided.
        metadata : dict, optional
            Additional metadata to store with the position
            
        Returns:
        --------
        bool : Whether the trade was executed successfully
        """
        optionid = option_data['optionid']
        
        # Get appropriate price - includes contract_size
        price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
            option_data, is_opening=True, is_long=True
        )
        
        if price_per_share <= 0:
            self.logger(f"Invalid price ({price_per_share}) for option {optionid}")
            return False
        
        # Calculate contracts from notional if provided
        if notional is not None and notional > 0:
            contracts = int(notional / price_per_contract)
            if contracts <= 0:
                self.logger(f"Notional amount ${notional:.2f} too small for option price ${price_per_contract:.2f}")
                return False
        
        if contracts <= 0:
            self.logger(f"Invalid contract count ({contracts}) for option {optionid}")
            return False
        
        # Calculate total cost
        total_cost = contracts * price_per_contract
        
        # Calculate commission - using actual contract size
        commission = self.commission_model.calculate_commission(
            contracts , price_per_share
        )
        total_cost += commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            self.logger(f"Insufficient cash (${self.cash:.2f}) for option trade (${total_cost:.2f})")
            
            # Try to calculate maximum affordable contracts
            max_contracts = int(self.cash / (price_per_contract + commission / contracts if contracts > 0 else 0))
            if max_contracts <= 0:
                return False
                
            self.logger(f"Adjusting position to {max_contracts} contracts based on available cash")
            contracts = max_contracts
            total_cost = contracts * price_per_contract + self.commission_model.calculate_commission(
                contracts , price_per_share
            )
        
        # Get the current cfadj value for future adjustment tracking
        current_cfadj = option_data.get('cfadj', 1.0)
        if pd.isna(current_cfadj) or current_cfadj <= 0:
            current_cfadj = 1.0
        
        # Execute trade
        if optionid in self.option_positions:
            # Update existing position
            old_basis = self.option_positions[optionid][0]
            old_contracts = self.option_positions[optionid][1]
            
            # Check if position is long
            if not self.option_positions[optionid][7]:  # is_long flag
                self.logger(f"Cannot add long position to existing short position for option {optionid}")
                return False
                
            new_contracts = old_contracts + contracts
            new_basis = ((old_basis * old_contracts) + (price_per_contract * contracts)) / new_contracts
            self.option_positions[optionid][0] = new_basis
            self.option_positions[optionid][1] = new_contracts
            self.option_positions[optionid][2] = price_per_contract  # Update current price
            
            # Update metadata if provided and position has metadata field
            if metadata is not None and len(self.option_positions[optionid]) > 10:
                self.option_positions[optionid][10].update(metadata)
        else:
            # Create position list
            position = [
                price_per_contract,  # cost_basis
                contracts,           # contracts
                price_per_contract,  # current_price
                date,                # entry_date
                option_data['exdate'], # expiry_date
                option_data['cp_flag'], # option_type
                option_data['strike_price'], # strike
                True,                # is_long flag
                contract_size,       # contract_size
                current_cfadj        # last_cfadj
            ]
            
            # Add metadata if provided
            if metadata is not None:
                position.append(metadata)
            
            # Create new position
            self.option_positions[optionid] = position
        
        # Update cash
        self.cash -= total_cost
        
        # Record trade
        self.trades_history.append({
            'date': date,
            'optionid': optionid,
            'secid': option_data.get('secid'),
            'permno': option_data.get('permno'),
            'action': 'OPEN_LONG',
            'contracts': contracts,
            'contract_size': contract_size,
            'price_per_share': price_per_share,
            'price_per_contract': price_per_contract,
            'commission': commission,
            'option_type': option_data['cp_flag'],
            'strike': option_data['strike_price'],
            'expiry_date': option_data['exdate'],
            'total_cost': total_cost,
            'cfadj': current_cfadj,
            'notional': notional
        })
        
        self.logger(f"Opened long position of {contracts} contracts for option {optionid} at ${price_per_contract:.2f} per contract")
        return True

    def open_short_position_by_notional(self, date, option_data, notional=None, contracts=None, metadata=None):
        """
        Open a short position using notional amount instead of weight
        
        Parameters:
        -----------
        date : str
            Trade date in 'YYYY-MM-DD' format
        option_data : Series or dict
            Option data
        notional : float, optional
            Notional amount to invest (e.g., $10,000). If None, uses contracts.
        contracts : int, optional
            Number of contracts to short. Ignored if notional is provided.
        metadata : dict, optional
            Additional metadata to store with the position
            
        Returns:
        --------
        bool : Whether the trade was executed successfully
        """
        optionid = option_data['optionid']
        
        # Get appropriate price
        price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
            option_data, is_opening=True, is_long=False
        )
        
        if price_per_share <= 0:
            self.logger(f"Invalid price ({price_per_share}) for option {optionid}")
            return False
        
        # Calculate contracts from notional if provided
        if notional is not None and notional > 0:
            contracts = int(notional / price_per_contract)
            if contracts <= 0:
                self.logger(f"Notional amount ${notional:.2f} too small for option price ${price_per_contract:.2f}")
                return False
        
        if contracts <= 0:
            self.logger(f"Invalid contract count ({contracts}) for option {optionid}")
            return False
        
        # Calculate proceeds per contract
        total_proceeds = contracts * price_per_contract
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(
            contracts , price_per_share
        )
        net_proceeds = total_proceeds - commission
        
        # Get the current cfadj value for future adjustment tracking
        current_cfadj = option_data.get('cfadj', 1.0)
        if pd.isna(current_cfadj) or current_cfadj <= 0:
            current_cfadj = 1.0
            
        # For short positions, store contract count as positive but flag as short
        if optionid in self.option_positions:
            # Update existing position
            old_basis = self.option_positions[optionid][0]
            old_contracts = self.option_positions[optionid][1]
            
            # Check if position is short
            if self.option_positions[optionid][7]:  # is_long flag
                self.logger(f"Cannot add short position to existing long position for option {optionid}")
                return False
                
            new_contracts = old_contracts + contracts
            new_basis = ((old_basis * old_contracts) + (price_per_contract * contracts)) / new_contracts
            self.option_positions[optionid][0] = new_basis
            self.option_positions[optionid][1] = new_contracts
            self.option_positions[optionid][2] = price_per_contract  # Update current price
            
            # Update metadata if provided and position has metadata field
            if metadata is not None and len(self.option_positions[optionid]) > 10:
                self.option_positions[optionid][10].update(metadata)
        else:
            # Create position list
            position = [
                price_per_contract,  # opening_price
                contracts,           # contracts
                price_per_contract,  # current_price
                date,                # entry_date
                option_data['exdate'], # expiry_date
                option_data['cp_flag'], # option_type
                option_data['strike_price'], # strike
                False,               # is_long flag = False for short positions
                contract_size,       # contract_size
                current_cfadj        # last_cfadj
            ]
            
            # Add metadata if provided
            if metadata is not None:
                position.append(metadata)
            
            # Create new short position
            self.option_positions[optionid] = position
        
        # Update cash (shorting adds cash)
        self.cash += net_proceeds
        
        # Record trade
        self.trades_history.append({
            'date': date,
            'optionid': optionid,
            'secid': option_data.get('secid'),
            'permno': option_data.get('permno'),
            'action': 'OPEN_SHORT',
            'contracts': contracts,
            'contract_size': contract_size,
            'price_per_share': price_per_share,
            'price_per_contract': price_per_contract,
            'commission': commission,
            'option_type': option_data['cp_flag'],
            'strike': option_data['strike_price'],
            'expiry_date': option_data['exdate'],
            'net_proceeds': net_proceeds,
            'cfadj': current_cfadj,
            'notional': notional
        })
        
        self.logger(f"Opened short position of {contracts} contracts for option {optionid} at ${price_per_contract:.2f} per contract")
        return True

    def close_short_position(self, date, option_data, contracts=None):
        """
        Close a short position for an option contract with better handling of insufficient cash
        
        Parameters:
        -----------
        date : str
            Trade date in 'YYYY-MM-DD' format
        option_data : Series or dict
            Option data
        contracts : int, optional
            Number of contracts to close (positive number). If None, closes all.
            
        Returns:
        --------
        bool : Whether the trade was executed successfully (even partially)
        """
        optionid = option_data['optionid']
        
        if optionid not in self.option_positions:
            self.logger(f"No position found for option {optionid}")
            return False
        
        # Verify this is a short position
        if self.option_positions[optionid][7]:  # is_long flag
            self.logger(f"Position for option {optionid} is not a short position")
            return False
            
        current_contracts = self.option_positions[optionid][1]
        contract_size = self.option_positions[optionid][8]
            
        if contracts is None:
            contracts = current_contracts
        elif contracts > current_contracts:
            contracts = current_contracts  # Adjust to available contracts
        
        # Get appropriate price for closing shorts (buying back)
        price_per_share, price_per_contract, _ = opt.calculate_option_price(
            option_data, is_opening=False, is_long=False
        )
        
        if price_per_share <= 0:
            self.logger(f"Invalid price ({price_per_share}) for option {optionid}")
            return False
        
        # Calculate cost per contract including commission
        commission_per_contract = self.commission_model.calculate_commission(
            1 , price_per_share
        )
        cost_per_contract = price_per_contract + commission_per_contract
        
        # Check if we have enough cash for all contracts
        total_cost = contracts * price_per_contract + self.commission_model.calculate_commission(
            contracts , price_per_share
        )
        
        # If not enough cash for all contracts, calculate how many we can afford
        if total_cost > self.cash:
            self.logger(f"Insufficient cash (${self.cash:.2f}) to close all {contracts} short contracts (${total_cost:.2f})")
            
            # Calculate max contracts we can afford (considering minimum commission)
            max_affordable = int(self.cash / cost_per_contract)
            
            if max_affordable <= 0:
                self.logger(f"Cannot afford to close any contracts at this time")
                return False
            
            # Adjust contracts to what we can afford
            contracts = max_affordable
            self.logger(f"Partially closing position - will close {contracts} out of {current_contracts} contracts")
            
            # Recalculate commission and total cost
            commission = self.commission_model.calculate_commission(
                contracts , price_per_share
            )
            total_cost = contracts * price_per_contract + commission
        else:
            # Calculate commission for full closing
            commission = self.commission_model.calculate_commission(
                contracts , price_per_share
            )
        
        # Calculate P&L
        opening_price = self.option_positions[optionid][0]
        pl_per_contract = opening_price - price_per_contract  # For shorts, profit is entry price minus exit price
        total_pl = pl_per_contract * contracts
        
        # Update position
        if contracts == current_contracts:
            # Close position
            del self.option_positions[optionid]
        else:
            # Reduce position
            self.option_positions[optionid][1] -= contracts
        
        # Update cash
        self.cash -= total_cost
        
        # Record trade
        self.trades_history.append({
            'date': date,
            'optionid': optionid,
            'secid': option_data.get('secid'),
            'permno': option_data.get('permno'),
            'action': 'CLOSE_SHORT',
            'contracts': contracts,
            'contract_size': contract_size,
            'price_per_share': price_per_share,
            'price_per_contract': price_per_contract,
            'commission': commission,
            'opening_price': opening_price,
            'pl_per_contract': pl_per_contract,
            'total_pl': total_pl,
            'option_type': option_data['cp_flag'],
            'strike': option_data['strike_price'],
            'expiry_date': option_data['exdate'],
            'total_cost': total_cost,
            'cfadj': option_data.get('cfadj', 1.0)
        })
        
        self.logger(f"Closed short position of {contracts} contracts for option {optionid} at ${price_per_contract:.2f} per contract (P&L: ${total_pl:.2f})")
        return True

    def process_option_targets(self, date, targets):
        """
        Modified process_option_targets to handle notional-based position sizing
        
        Parameters:
        -----------
        date : str
            Current date in 'YYYY-MM-DD' format
        targets : dict
            Dictionary of option targets from strategy
            
        Returns:
        --------
        dict : Result summary
        """
        if not targets:
            return {'trades': 0, 'success': 0}
        
        # First, check if this is a rebalance day
        rebalance = targets.get('rebalance', False)
        
        # Get current portfolio value for weight calculations
        portfolio_value = self.get_portfolio_value()
        
        # Track trades
        trade_count = 0
        success_count = 0
        long_value = 0.0
        short_value = 0.0
        
        # Check for positions to close based on target date
        current_date = pd.to_datetime(date)
        for optionid in list(self.option_positions.keys()):
            position = self.option_positions[optionid]
            is_long = position[7]  # Is long flag
            
            # Check if position has a target close date from metadata
            position_metadata = position[10] if len(position) > 10 else {}
            if isinstance(position_metadata, dict) and 'target_date' in position_metadata:
                target_date = pd.to_datetime(position_metadata['target_date'])
                
                # If target date reached, close the position
                if current_date >= target_date:
                    self.logger(f"Closing position for option {optionid} - reached target holding period")
                    
                    # Get current option data
                    if self.option_data_dic is not None:
                        option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
                    else:
                        option_data = opt.load_option_by_id(date, optionid, self.db_path)
                    
                    if option_data is not None:
                        trade_count += 1
                        
                        if is_long:
                            success = self.close_long_position(date, option_data)
                        else:
                            success = self.close_short_position(date, option_data)
                            
                        if success:
                            success_count += 1
                            contracts = position[1]
                            current_price = position[2]
                            if is_long:
                                long_value += contracts * current_price
                            else:
                                short_value += contracts * current_price
        
        # If rebalancing, close all remaining positions
        if rebalance:
            # Get all optionids from current positions
            optionids = list(self.option_positions.keys())
            
            for optionid in optionids:
                position = self.option_positions[optionid]
                is_long = position[7]
                contracts = position[1]
                current_price = position[2]
                
                if self.option_data_dic is not None:
                    option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
                else:
                    option_data = opt.load_option_by_id(date, optionid, self.db_path)
                
                if option_data is not None:
                    trade_count += 1
                    
                    if is_long:
                        success = self.close_long_position(date, option_data)
                        if success:
                            success_count += 1
                            long_value += contracts * current_price
                    else:
                        success = self.close_short_position(date, option_data)
                        if success:
                            success_count += 1
                            short_value += contracts * current_price
                else:
                    self.logger(f"Warning: Cannot close position for option {optionid} - no data available")
        
        # Process long calls
        for option in targets.get('long_calls', []):
            optionid = option.get('optionid')
            notional = option.get('notional')  # Use notional instead of weight
            target_date = option.get('target_date')  # When to close this position
            
            if self.option_data_dic is not None:
                option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
            else:
                option_data = opt.load_option_by_id(date, optionid, self.db_path)
            
            if option_data is not None:
                trade_count += 1
                
                # Store metadata for future reference
                metadata = {
                    'target_date': target_date,
                    'strategy': option.get('strategy', 'long_call')
                }
                
                # Use notional-based position opening
                if notional is not None:
                    success = self.open_long_position_by_notional(date, option_data, notional=notional, metadata=metadata)
                    if success:
                        success_count += 1
                        long_value += notional
                else:
                    # Fall back to weight-based for backwards compatibility
                    weight = option.get('weight', 0.01)
                    success = self.open_long_position(date, option_data, weight=weight, metadata=metadata)
                    if success:
                        success_count += 1
                        
                        # Calculate and update contracts in target
                        price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
                            option_data, is_opening=True, is_long=True
                        )
                        contracts = opt.calculate_contracts(weight, price_per_share, portfolio_value, contract_size)
                        option['contracts'] = contracts
                        long_value += contracts * price_per_contract
        
        # Process long puts
        for option in targets.get('long_puts', []):
            optionid = option.get('optionid')
            notional = option.get('notional')  # Use notional instead of weight
            target_date = option.get('target_date')  # When to close this position
            
            if self.option_data_dic is not None:
                option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
            else:
                option_data = opt.load_option_by_id(date, optionid, self.db_path)
            
            if option_data is not None:
                trade_count += 1
                
                # Store metadata for future reference
                metadata = {
                    'target_date': target_date,
                    'strategy': option.get('strategy', 'long_put')
                }
                
                # Use notional-based position opening
                if notional is not None:
                    success = self.open_long_position_by_notional(date, option_data, notional=notional, metadata=metadata)
                    if success:
                        success_count += 1
                        long_value += notional
                else:
                    # Fall back to weight-based for backwards compatibility
                    weight = option.get('weight', 0.01)
                    success = self.open_long_position(date, option_data, weight=weight, metadata=metadata)
                    if success:
                        success_count += 1
                        
                        # Calculate and update contracts in target
                        price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
                            option_data, is_opening=True, is_long=True
                        )
                        contracts = opt.calculate_contracts(weight, price_per_share, portfolio_value, contract_size)
                        option['contracts'] = contracts
                        long_value += contracts * price_per_contract
        
        # Process short calls
        for option in targets.get('short_calls', []):
            optionid = option.get('optionid')
            notional = option.get('notional')  # Use notional instead of weight
            target_date = option.get('target_date')  # When to close this position
            
            if self.option_data_dic is not None:
                option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
            else:
                option_data = opt.load_option_by_id(date, optionid, self.db_path)
            
            if option_data is not None:
                trade_count += 1
                
                # Store metadata for future reference
                metadata = {
                    'target_date': target_date,
                    'strategy': option.get('strategy', 'short_call')
                }
                
                # Use notional-based position opening
                if notional is not None:
                    success = self.open_short_position_by_notional(date, option_data, notional=notional, metadata=metadata)
                    if success:
                        success_count += 1
                        short_value += notional
                else:
                    # Fall back to weight-based for backwards compatibility
                    weight = option.get('weight', 0.01)
                    success = self.open_short_position(date, option_data, weight=weight, metadata=metadata)
                    if success:
                        success_count += 1
                        
                        # Calculate and update contracts in target
                        price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
                            option_data, is_opening=True, is_long=False
                        )
                        contracts = opt.calculate_contracts(weight, price_per_share, portfolio_value, contract_size)
                        option['contracts'] = contracts
                        short_value += contracts * price_per_contract
        
        # Process short puts
        for option in targets.get('short_puts', []):
            optionid = option.get('optionid')
            notional = option.get('notional')  # Use notional instead of weight
            target_date = option.get('target_date')  # When to close this position
            
            if self.option_data_dic is not None:
                option_data = opt.load_option_by_id(date, optionid, self.db_path, self.option_data_dic)
            else:
                option_data = opt.load_option_by_id(date, optionid, self.db_path)
            
            if option_data is not None:
                trade_count += 1
                
                # Store metadata for future reference
                metadata = {
                    'target_date': target_date,
                    'strategy': option.get('strategy', 'short_put')
                }
                
                # Use notional-based position opening
                if notional is not None:
                    success = self.open_short_position_by_notional(date, option_data, notional=notional, metadata=metadata)
                    if success:
                        success_count += 1
                        short_value += notional
                else:
                    # Fall back to weight-based for backwards compatibility
                    weight = option.get('weight', 0.01)
                    success = self.open_short_position(date, option_data, weight=weight, metadata=metadata)
                    if success:
                        success_count += 1
                        
                        # Calculate and update contracts in target
                        price_per_share, price_per_contract, contract_size = opt.calculate_option_price(
                            option_data, is_opening=True, is_long=False
                        )
                        contracts = opt.calculate_contracts(weight, price_per_share, portfolio_value, contract_size)
                        option['contracts'] = contracts
                        short_value += contracts * price_per_contract
        
        # Calculate total trade value for turnover calculation
        total_value = long_value + short_value
        
        return {
            'trades': trade_count,
            'success': success_count,
            'rebalance': rebalance,
            'long_value': long_value,
            'short_value': short_value,
            'total_value': total_value
        }




class OptionBacktest:
    """
    Class for backtesting option trading strategies.
    Follows flow of stock backtesting with trades at market close.
    """
    def __init__(self, cash, commission_model, data_df_dic, trading_dates, 
             db_path, strategy, option_data_dic=None, dir_='option_results',
             gvkeyx='000003', save_format='png'):
        """
        Initialize option backtest system with preloaded option data
        
        Parameters:
        -----------
        cash : float
            Initial cash
        commission_model : CommissionModel
            Commission model for option trades
        data_df_dic : dict
            Dictionary of daily data {date: DataFrame}
        trading_dates : list
            List of trading dates
        db_path : str
            Path to database file
        strategy : OptionStrategyBase
            Option strategy object that generates targets
        option_data_dic : dict, optional
            Preloaded option data {date: DataFrame}
        dir_ : str
            Directory for saving results
        gvkeyx : str
            Benchmark index code
        save_format : str
            Format for saving results
        """
        # Create output directory
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        
        # Initialize file paths
        self.dir_ = dir_
        self.save_format = save_format
        self.return_df_path = os.path.join(dir_, 'option_return_df.csv')
        
        # Store data
        self.data_df_dic = data_df_dic
        self.trading_dates = trading_dates
        
        self.db_path = db_path
        self.strategy = strategy
        self.gvkeyx = gvkeyx
        
        # Store preloaded option data if provided
        self.option_data_dic = option_data_dic
        
        # Initialize backtest logging
        self.txt_file = open(os.path.join(dir_, 'option_backtest_logs.txt'), 'w+')
        
        # Trading log for the OptionTrading class
        trading_log_path = os.path.join(dir_, 'option_trading_logs.txt')
        
        # Initialize options trading system
        self.trading = OptionTrading(
            cash=cash,
            commission_model=commission_model,
            db_path=db_path,
            is_print=False,
            txt_file=open(trading_log_path, 'w+'),
            option_data_dic=option_data_dic
        )
        
        # If option data is preloaded, modify the option data access methods
        #if self.option_data_dic is not None:
        #    self._setup_preloaded_data_methods()
        
        # Performance tracking
        self.backtest_data = {
            'portfolio_values': [cash],
            'benchmark_values': [],
            'daily_returns': [0.0],
            'benchmark_returns': [],
            'trade_count': 0,
            'turnover_values': [],
            'component_pnl': {
                'long_calls': [0.0],
                'long_puts': [0.0],
                'short_calls': [0.0],
                'short_puts': [0.0],
                'total': [0.0]
            }
        }
        
        # Dictionary to store daily trade details
        self.daily_trade_details = {}
        
        # For turnover calculation
        self.turnover_dic = {}
        
        # Load benchmark data
        self._load_benchmark_data()


    # 2. Add a method to set up the preloaded data handling
    def _setup_preloaded_data_methods(self):
        """
        Override option data access methods to use preloaded data
        """
        import types
        
        # Store original methods for fallback
        self._original_load_option_by_id = opt.load_option_by_id
        self._original_load_option_batch = opt.load_option_batch
        
        # Modify OptionTrading methods to use preloaded data
        self.trading.load_option_by_id = types.MethodType(
            lambda self, date, optionid: opt.load_option_by_id_from_preloaded(
                self.option_data_dic, date, optionid, self.db_path
            ), 
            self.trading
        )
        
        self.trading.load_option_batch = types.MethodType(
            lambda self, date, optionids: opt.load_option_batch_from_preloaded(
                self.option_data_dic, date, optionids, self.db_path
            ), 
            self.trading
        )
        
        # Log that we're using preloaded data
        self.logger("Using preloaded option data for backtest")    



    def logger(self, txt_str):
        """Log message to file"""
        if self.txt_file is not None and not self.txt_file.closed:
            print(txt_str, file=self.txt_file)
            
    def _load_benchmark_data(self):
        """
        Load benchmark data and initialize benchmark-related class members
        """
        benchmark_file = os.path.join('Data_all', 'Benchmark_data', self.gvkeyx, f'benchmark_{self.gvkeyx}.parquet')
        if not os.path.exists(benchmark_file):
            self.logger(f"Benchmark file not found: {benchmark_file}")
            self.benchmark_returns = None
            self.benchmark_prices = None
            self.benchmark_cum = None
            self.daily_rf_rates = None
            return
            
        # Read benchmark data
        benchmark_df = pd.read_parquet(benchmark_file)
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        benchmark_df.set_index('date', inplace=True)
        
        # Get benchmark returns and prices for our trading dates
        trading_dates = pd.to_datetime(self.trading_dates)
        # Filter to only those dates in the benchmark data
        valid_dates = trading_dates[trading_dates.isin(benchmark_df.index)]
        
        if len(valid_dates) > 0:
            self.benchmark_returns = benchmark_df.loc[valid_dates, 'ret_idx']
            self.benchmark_prices = benchmark_df.loc[valid_dates, 'tr_idx']
            
            # Calculate cumulative returns relative to first date's price
            base_price = self.benchmark_prices.iloc[0]
            self.benchmark_cum = self.benchmark_prices / base_price
        else:
            self.logger("No benchmark data available for trading dates")
            self.benchmark_returns = None
            self.benchmark_prices = None
            self.benchmark_cum = None
        
        # Load risk-free rates
        rf_file = os.path.join('Data_all', 'RF_data', 'rf_rates.parquet')
        if os.path.exists(rf_file):
            rf_df = pd.read_parquet(rf_file)
            rf_df['date'] = pd.to_datetime(rf_df['date'])
            rf_df.set_index('date', inplace=True)
            
            # Get risk-free rates for our trading dates
            self.daily_rf_rates = rf_df.loc[valid_dates, 'rf'] if len(valid_dates) > 0 else None
        else:
            self.logger("Risk-free rate file not found")
            self.daily_rf_rates = None

    # 3. Create a new helper method to update the strategy with preloaded option handling
    def update_strategy_for_preloaded_data(self, strategy):
        """
        Update the strategy to use preloaded option data
        
        Parameters:
        -----------
        strategy : OptionStrategyBase
            Strategy object to update
        """
        import types
        
        # Store the strategy
        self.strategy = strategy
        
        # First check if we have preloaded data
        if self.option_data_dic is None:
            self.logger("No preloaded option data available for strategy")
            return
        
        # Create wrapped method to handle option data access in strategy
        def wrapped_get_option_data_for_stock(self, date_str, permno, db_path, **kwargs):
            """Wrapper to use preloaded data for option access in strategy"""
            import pandas as pd
            
            # Convert date for lookup
            date_obj = pd.to_datetime(date_str)
            
            # Check if we have data for this date
            if date_obj not in self.option_data_dic:
                # Fall back to original method
                from option_tools import get_option_data_for_stock
                return get_option_data_for_stock(date_str, permno, db_path, **kwargs)
            
            # Get option data for this date
            date_options = self.option_data_dic[date_obj]
            
            # Filter by permno
            if len(date_options) > 0:
                permno_options = date_options[date_options['permno'] == permno].copy()
                
                # Apply additional filters from kwargs if needed
                if 'days_to_expiry_range' in kwargs:
                    min_days, max_days = kwargs['days_to_expiry_range']
                    current_date = pd.to_datetime(date_str)
                    
                    # Calculate days to expiry
                    permno_options['days_to_expiry'] = (pd.to_datetime(permno_options['exdate']) - current_date).dt.days
                    
                    # Filter by days to expiry
                    permno_options = permno_options[
                        (permno_options['days_to_expiry'] >= min_days) & 
                        (permno_options['days_to_expiry'] <= max_days)
                    ]
                
                # Filter by open interest
                if 'min_open_interest' in kwargs:
                    min_oi = kwargs['min_open_interest']
                    permno_options = permno_options[permno_options['open_interest'] >= min_oi]
                
                # Filter by option type
                if 'option_type' in kwargs:
                    opt_type = kwargs['option_type']
                    if opt_type in ['C', 'P']:
                        permno_options = permno_options[permno_options['cp_flag'] == opt_type]
                
                # Filter by delta ranges
                if 'delta_ranges' in kwargs and kwargs['delta_ranges'] is not None:
                    delta_ranges = kwargs['delta_ranges']
                    filtered_options = []
                    
                    for opt_type, (min_delta, max_delta) in delta_ranges.items():
                        type_options = permno_options[permno_options['cp_flag'] == opt_type]
                        delta_filtered = type_options[
                            (type_options['delta'] >= min_delta) & 
                            (type_options['delta'] <= max_delta)
                        ]
                        filtered_options.append(delta_filtered)
                    
                    if filtered_options:
                        permno_options = pd.concat(filtered_options, ignore_index=True)
                
                return permno_options
            
            # Fall back to original method if no options found
            from option_tools import get_option_data_for_stock
            return get_option_data_for_stock(date_str, permno, db_path, **kwargs)
        
        # Attach wrapped method to strategy
        strategy._get_option_data_for_stock = types.MethodType(
            wrapped_get_option_data_for_stock, 
            strategy
        )
        
        # Override get_option_data_for_stock method in strategy if it exists
        if hasattr(strategy, 'get_option_data_for_stock'):
            strategy.get_option_data_for_stock = strategy._get_option_data_for_stock
        
        # Log that we've updated the strategy
        self.logger(f"Updated strategy {strategy.__class__.__name__} to use preloaded option data")


    def before_market_close(self, date, current_data_df):
        """
        Process events and generate option targets before market close
        
        Parameters:
        -----------
        date : datetime
            Current trading date
        current_data_df : DataFrame
            Current day's stock data
            
        Returns:
        --------
        dict : Option targets
        """
        try:
            # Format date
            date_str = date.strftime('%Y-%m-%d')
            
            # 1. First handle option expiry for current date
            self.trading.handle_option_expiry(date_str)
            
            # 2. Handle corporate actions for existing positions
            #self.trading.handle_corporate_actions(date_str)
            
            
            # Generate option targets using strategy
            targets = self.strategy.generate_option_targets(
                date=date,
                data_df=current_data_df,
                db_path=self.db_path,  # Pass the temporary connection
                logger=self.logger
            )
            
            return targets
        except Exception as e:
            self.logger(f"Error in before_market_close for {date}: {str(e)}")
            import traceback
            self.logger(traceback.format_exc())
            return {}
            
    def market_close(self, targets, current_date, current_data_df):
            """
            Enhanced market_close method that also tracks component P&L
            
            Parameters:
            -----------
            targets : dict
                Option targets from strategy
            current_date : datetime
                Current trading date
            current_data_df : DataFrame
                Current day's stock data
                
            Returns:
            --------
            dict : Execution results
            """
            try:
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Get portfolio value before trading
                portfolio_value_before = self.trading.get_portfolio_value()
                
                # Update option prices first
                self.trading.update_option_prices(date_str)
                
                # Get component values before trading
                prev_component_values = self.trading.get_component_values()
                
                # Execute trades based on targets
                execution_result = self.trading.process_option_targets(date_str, targets)
                
                # Get trade details
                trades_attempted = execution_result.get('trades', 0)
                trades_executed = execution_result.get('success', 0)
                long_value = execution_result.get('long_value', 0.0)
                short_value = execution_result.get('short_value', 0.0)
                total_trade_value = execution_result.get('total_value', 0.0)
                
                # Get portfolio value after trading
                portfolio_value_after = self.trading.get_portfolio_value()
                
                # Calculate daily return
                if len(self.backtest_data['portfolio_values']) > 0:
                    prev_value = self.backtest_data['portfolio_values'][-1]
                    daily_return = (portfolio_value_after / prev_value) - 1 if prev_value > 0 else 0
                else:
                    daily_return = 0
                
                # Get component values after trading
                current_component_values = self.trading.get_component_values()
                
                # Calculate component P&L
                component_pnl = {}
                for component, prev_value in prev_component_values.items():
                    curr_value = current_component_values.get(component, 0.0)
                    component_pnl[component] = curr_value - prev_value
                
                # Update component P&L in backtest data
                for component, pnl in component_pnl.items():
                    self.backtest_data['component_pnl'][component].append(pnl)
                
                # Calculate total component P&L
                total_component_pnl = sum(component_pnl.values())
                self.backtest_data['component_pnl']['total'].append(total_component_pnl)
                
                # Store daily metrics
                self.backtest_data['portfolio_values'].append(portfolio_value_after)
                self.backtest_data['daily_returns'].append(daily_return)
                self.backtest_data['trade_count'] += trades_executed
                
                # Calculate turnover (similar to stock backtest)
                if portfolio_value_after > 0:
                    turnover = 100 * total_trade_value / 2 / portfolio_value_after
                else:
                    turnover = 0
                    
                # Store turnover
                self.turnover_dic[current_date] = turnover
                self.backtest_data['turnover_values'].append(turnover)
                
                # Store detailed trade info
                self.daily_trade_details[current_date] = {
                    'trades_attempted': trades_attempted,
                    'trades_executed': trades_executed,
                    'long_value': long_value,
                    'short_value': short_value,
                    'total_trade_value': total_trade_value,
                    'portfolio_value': portfolio_value_after,
                    'daily_return': daily_return,
                    'turnover': turnover,
                    'component_pnl': component_pnl
                }
                
                # Log results
                if trades_attempted > 0:
                    self.logger(f"\nTrading results for {date_str}:")
                    self.logger(f"Trades attempted: {trades_attempted}")
                    self.logger(f"Trades executed: {trades_executed}")
                    self.logger(f"Long value: ${long_value:,.2f}")
                    self.logger(f"Short value: ${short_value:,.2f}")
                    self.logger(f"Total trade value: ${total_trade_value:,.2f}")
                    self.logger(f"Turnover: {turnover:.2f}%")
                    
                    # Log component P&L
                    self.logger("\nComponent P&L:")
                    for component, pnl in component_pnl.items():
                        self.logger(f"{component}: ${pnl:.2f}")
                    self.logger(f"Total P&L: ${total_component_pnl:.2f}")
                    
                # Print position details
                self.print_position_details(date_str)
                
                # Log portfolio summary
                self.logger(f"\nPortfolio Summary for {date_str}:")
                self.logger(f"Cash: ${self.trading.cash:.2f}")
                self.logger(f"Portfolio Value: ${portfolio_value_after:.2f}")
                self.logger(f"Daily Return: {daily_return*100:.2f}%")
                self.logger(f"Option Positions: {len(self.trading.option_positions)}")
                
                return self.daily_trade_details[current_date]
                
            except Exception as e:
                self.logger(f"Error in market_close for {current_date}: {str(e)}")
                import traceback
                self.logger(traceback.format_exc())
                return {
                    'trades_attempted': 0,
                    'trades_executed': 0,
                    'portfolio_value': self.trading.get_portfolio_value(),
                    'daily_return': 0,
                    'turnover': 0,
                    'component_pnl': {'long_calls': 0, 'long_puts': 0, 'short_calls': 0, 'short_puts': 0, 'total': 0}
                }
            
    def print_position_details(self, date_str):
        """
        Print detailed position information
        
        Parameters:
        -----------
        date_str : str
            Current date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        None
        """
        if not self.trading.option_positions:
            self.logger("No active option positions")
            return
            
        self.logger("\nActive Option Positions:")
        self.logger(f"{'Option ID':<12} {'Type':<6} {'Strike':<10} {'Expiry':<12} {'Position':<10} {'Cost':<10} {'Current':<10} {'P&L':<10} {'cfadj':<6}")
        self.logger("-" * 90)
        
        total_pl = 0
        
        for optionid, position in self.trading.option_positions.items():
            # [cost_basis, contracts, current_price, entry_date, expiry_date, option_type, strike, is_long, contract_size, last_cfadj]
            cost_basis = position[0]
            contracts = position[1]
            current_price = position[2]
            expiry_date = position[4]
            option_type = position[5]
            strike = position[6]
            is_long = position[7]
            cfadj = position[9] if len(position) > 9 else 1.0
            
            # Position direction
            direction = "Long" if is_long else "Short"
            
            # Calculate P&L
            if is_long:
                position_pl = (current_price - cost_basis) * contracts
            else:
                position_pl = (cost_basis - current_price) * contracts
                
            total_pl += position_pl
            
            # Format expiry date
            if isinstance(expiry_date, str):
                expiry_str = expiry_date
            else:
                expiry_str = expiry_date.strftime('%Y-%m-%d')
                
            # Format outputs
            self.logger(f"{optionid:<12} {option_type:<6} {strike:<10.2f} {expiry_str:<12} "
                       f"{direction+' '+str(contracts):<10} {cost_basis:<10.2f} {current_price:<10.2f} "
                       f"{position_pl:<10.2f} {cfadj:<6.2f}")
            
        self.logger("-" * 90)
        self.logger(f"Total unrealized P&L: ${total_pl:.2f}")
            
    def backtest(self):
        """
        Run the option backtest
        
        Returns:
        --------
        dict : Backtest results
        """
        try:
            self.logger("Starting option backtest...")
            
            # Reset tracking variables except the initial cash value
            initial_cash = self.backtest_data['portfolio_values'][0]

                    # Performance tracking
            self.backtest_data = {
                'portfolio_values': [initial_cash],
                'benchmark_values': [],
                'daily_returns': [0.0],
                'benchmark_returns': [],
                'trade_count': 0,
                'turnover_values': [],
                'component_pnl': {
                    'long_calls': [0.0],
                    'long_puts': [0.0],
                    'short_calls': [0.0],
                    'short_puts': [0.0],
                    'total': [0.0]
                }
            }
            
            # Reset trading system
            self.trading.cash = initial_cash
            self.trading.option_positions = {}
            self.trading.trades_history = []
            self.trading.expired_options = []
            self.trading.corporate_action_history = []
            
            # Store benchmark starting value if available
            if self.benchmark_prices is not None and len(self.benchmark_prices) > 0:
                self.backtest_data['benchmark_values'].append(self.benchmark_prices.iloc[0])
            
            # Process each trading date
            for i in tqdm(range(1, len(self.trading_dates)), desc="Processing dates"):
                # Get current date
                current_date = self.trading_dates[i]
                
                # Get stock data for current date
                current_data_df = self.data_df_dic[current_date]
                
                # Generate option targets before market close
                targets = self.before_market_close(current_date, current_data_df)
                
                # Execute trades at market close
                result = self.market_close(targets, current_date, current_data_df)
                
                # Add benchmark value if available
                if self.benchmark_prices is not None:
                    date_str = current_date.strftime('%Y-%m-%d')
                    benchmark_idx = self.benchmark_prices.index.get_indexer([pd.to_datetime(date_str)], method='nearest')[0]
                    if benchmark_idx >= 0 and benchmark_idx < len(self.benchmark_prices):
                        self.backtest_data['benchmark_values'].append(self.benchmark_prices.iloc[benchmark_idx])
                        
                        # Calculate benchmark return
                        if len(self.backtest_data['benchmark_values']) > 1:
                            prev_benchmark = self.backtest_data['benchmark_values'][-2]
                            benchmark_return = self.backtest_data['benchmark_values'][-1] / prev_benchmark - 1
                            self.backtest_data['benchmark_returns'].append(benchmark_return)
                        else:
                            self.backtest_data['benchmark_returns'].append(0)
            
            # Calculate final performance metrics
            self.calculate_performance_metrics()
            
            # Save results
            self.save_results()
            
            self.logger("Backtest completed.")
            return self.backtest_data
            
        except Exception as e:
            self.logger(f"Error in backtest: {str(e)}")
            import traceback
            self.logger(traceback.format_exc())
            return {}
        finally:
            # Close log files
            if self.txt_file is not None and not self.txt_file.closed:
                self.txt_file.close()
            if self.trading.txt_file is not None and not self.trading.txt_file.closed:
                self.trading.txt_file.close()
                
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics from raw data
        
        Returns:
        --------
        dict : Performance metrics
        """
        # Convert to numpy arrays for faster calculations
        portfolio_values = np.array(self.backtest_data['portfolio_values'])
        daily_returns = np.array(self.backtest_data['daily_returns'])
        
        # Calculate basic return metrics
        start_value = portfolio_values[0]
        end_value = portfolio_values[-1]
        total_return = end_value / start_value - 1
        
        # Calculate annualized metrics
        trading_days_per_year = 252
        num_days = len(daily_returns)
        years = num_days / trading_days_per_year
        
        # Annualized return
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate risk metrics
        if len(daily_returns) > 1:
            # Volatility
            volatility = np.std(daily_returns) * np.sqrt(trading_days_per_year)
            
            # Calculate Sharpe ratio if we have risk-free rates
            if self.daily_rf_rates is not None:
                # Use only the needed number of risk-free rates
                rf_rates = self.daily_rf_rates[:len(daily_returns)].values if len(self.daily_rf_rates) >= len(daily_returns) else np.zeros(len(daily_returns))
                excess_returns = daily_returns - rf_rates
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days_per_year) if np.std(excess_returns) > 0 else 0
            else:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(trading_days_per_year) if np.std(daily_returns) > 0 else 0
                
            # Calculate drawdown
            cum_returns = np.cumprod(1 + daily_returns)
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / peak
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Calculate benchmark comparison metrics
        if 'benchmark_returns' in self.backtest_data and len(self.backtest_data['benchmark_returns']) > 0:
            benchmark_returns = np.array(self.backtest_data['benchmark_returns'])
            
            # Tracking error
            if len(benchmark_returns) == len(daily_returns[1:]):  # Skip first day return which is 0
                tracking_error = np.std(daily_returns[1:] - benchmark_returns) * np.sqrt(trading_days_per_year)
                
                # Information ratio
                excess_returns_benchmark = daily_returns[1:] - benchmark_returns
                information_ratio = np.mean(excess_returns_benchmark) / np.std(excess_returns_benchmark) * np.sqrt(trading_days_per_year) if np.std(excess_returns_benchmark) > 0 else 0
                
                # Beta
                covariance = np.cov(daily_returns[1:], benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Alpha (annualized)
                benchmark_ann_return = (1 + np.prod(1 + benchmark_returns) - 1) ** (1 / years) - 1 if years > 0 else 0
                alpha = ann_return - (self.daily_rf_rates.mean() * trading_days_per_year + beta * (benchmark_ann_return - self.daily_rf_rates.mean() * trading_days_per_year)) if self.daily_rf_rates is not None else 0
            else:
                tracking_error = 0
                information_ratio = 0
                beta = 0
                alpha = 0
        else:
            tracking_error = 0
            information_ratio = 0
            beta = 0
            alpha = 0
        
        # Calculate average turnover
        avg_turnover = np.mean(self.backtest_data['turnover_values']) if self.backtest_data['turnover_values'] else 0
        
        # Store results
        self.performance_metrics = {
            'total_return': total_return * 100,  # Convert to percentage
            'annualized_return': ann_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'tracking_error': tracking_error * 100,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha * 100,
            'trades': self.backtest_data['trade_count'],
            'final_portfolio_value': end_value,
            'profit': end_value - start_value,
            'avg_turnover': avg_turnover
        }
        
        return self.performance_metrics
        
    def save_results(self):
        """
        Enhanced save_results method that also saves component P&L
        
        Returns:
        --------
        None
        """
        # Save returns to CSV
        returns_df = pd.DataFrame({
            'date': self.trading_dates[:len(self.backtest_data['portfolio_values'])],
            'portfolio_value': self.backtest_data['portfolio_values'],
            'daily_return': self.backtest_data['daily_returns']
        })
        
        # Add benchmark values if available
        if 'benchmark_values' in self.backtest_data and len(self.backtest_data['benchmark_values']) > 0:
            returns_df['benchmark_value'] = self.backtest_data['benchmark_values'] + [None] * (len(returns_df) - len(self.backtest_data['benchmark_values']))
        
        if 'benchmark_returns' in self.backtest_data and len(self.backtest_data['benchmark_returns']) > 0:
            returns_df['benchmark_return'] = [0] + self.backtest_data['benchmark_returns'] + [None] * (len(returns_df) - len(self.backtest_data['benchmark_returns']) - 1)
        
        # Add component P&L
        for component, pnl_list in self.backtest_data['component_pnl'].items():
            returns_df[f'pnl_{component}'] = pnl_list + [None] * (len(returns_df) - len(pnl_list))
        
        # Add turnover if available
        if self.turnover_dic:
            turnover_series = pd.Series(self.turnover_dic)
            turnover_df = pd.DataFrame({
                'date': turnover_series.index,
                'turnover': turnover_series.values
            })
            returns_df = returns_df.merge(turnover_df, on='date', how='left')
        
        # Save to CSV
        returns_df.to_csv(self.return_df_path, index=False)
        
        # Save trade history if available
        if hasattr(self.trading, 'trades_history') and self.trading.trades_history:
            trade_df = pd.DataFrame(self.trading.trades_history)
            trade_path = os.path.join(self.dir_, 'option_trades.csv')
            trade_df.to_csv(trade_path, index=False)
        
        # Save expired options if available
        if hasattr(self.trading, 'expired_options') and self.trading.expired_options:
            expire_df = pd.DataFrame(self.trading.expired_options)
            expire_path = os.path.join(self.dir_, 'option_expirations.csv')
            expire_df.to_csv(expire_path, index=False)
        
        # Save corporate action history if available
        if hasattr(self.trading, 'corporate_action_history') and self.trading.corporate_action_history:
            ca_df = pd.DataFrame(self.trading.corporate_action_history)
            ca_path = os.path.join(self.dir_, 'option_corporate_actions.csv')
            ca_df.to_csv(ca_path, index=False)
        
        # Save performance metrics
        if hasattr(self, 'performance_metrics'):
            metrics_df = pd.DataFrame([self.performance_metrics])
            metrics_path = os.path.join(self.dir_, 'option_performance.csv')
            metrics_df.to_csv(metrics_path, index=False)
            
        # Save component P&L summary
        component_pnl_df = pd.DataFrame({
            'component': list(self.backtest_data['component_pnl'].keys()),
            'total_pnl': [sum(pnl_list) for pnl_list in self.backtest_data['component_pnl'].values()],
            'mean_daily_pnl': [np.mean(pnl_list) for pnl_list in self.backtest_data['component_pnl'].values()],
            'std_daily_pnl': [np.std(pnl_list) for pnl_list in self.backtest_data['component_pnl'].values()]
        })
        component_pnl_path = os.path.join(self.dir_, 'component_pnl_summary.csv')
        component_pnl_df.to_csv(component_pnl_path, index=False)
            
        self.logger(f"Results saved to {self.dir_}")
    '''
    def plot_performance(self, figsize=(16, 12)):
        """
        Enhanced plot_performance method that also plots component P&L
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        
        # Create a figure with 3 subplots (main performance, daily returns, component P&L)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Convert dates to proper format
        dates = pd.to_datetime(self.trading_dates[:len(self.backtest_data['portfolio_values'])])
        
        # Plot portfolio values
        portfolio_values = np.array(self.backtest_data['portfolio_values'])
        
        # Calculate relative returns for better visualization
        relative_portfolio_values = portfolio_values / portfolio_values[0]
        
        portfolio_line, = ax1.plot(dates, relative_portfolio_values, label='Option Strategy', linewidth=2)
        
        # Add benchmark if available
        if 'benchmark_values' in self.backtest_data and len(self.backtest_data['benchmark_values']) > 1:
            # Rescale benchmark to same starting value for comparison
            benchmark_values = np.array(self.backtest_data['benchmark_values'])
            relative_benchmark = benchmark_values / benchmark_values[0]
            
            benchmark_line, = ax1.plot(
                dates[:len(relative_benchmark)],
                relative_benchmark,
                label='Benchmark',
                linestyle='--'
            )
            
            # Add legend with performance metrics
            if hasattr(self, 'performance_metrics'):
                portfolio_label = f'Option Strategy (Ann. Return: {self.performance_metrics["annualized_return"]:.2f}%, Sharpe: {self.performance_metrics["sharpe_ratio"]:.2f})'
                benchmark_ann_return = (benchmark_values[-1] / benchmark_values[0]) ** (252 / len(benchmark_values)) - 1
                benchmark_label = f'Benchmark (Ann. Return: {benchmark_ann_return*100:.2f}%)'
                
                portfolio_line.set_label(portfolio_label)
                benchmark_line.set_label(benchmark_label)
        
        ax1.set_title('Option Strategy Performance')
        ax1.set_ylabel('Relative Value (1.0 = Initial)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot daily returns
        daily_returns = np.array(self.backtest_data['daily_returns'])
        ax2.bar(dates, daily_returns * 100)  # Convert to percentage
        
        ax2.set_ylabel('Daily Return (%)')
        ax2.grid(True)
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Plot component P&L
        # Create DataFrame for component P&L with dates
        component_pnl_df = pd.DataFrame({
            'date': dates,
            'long_calls': self.backtest_data['component_pnl']['long_calls'] + [0] * (len(dates) - len(self.backtest_data['component_pnl']['long_calls'])),
            'long_puts': self.backtest_data['component_pnl']['long_puts'] + [0] * (len(dates) - len(self.backtest_data['component_pnl']['long_puts'])),
            'short_calls': self.backtest_data['component_pnl']['short_calls'] + [0] * (len(dates) - len(self.backtest_data['component_pnl']['short_calls'])),
            'short_puts': self.backtest_data['component_pnl']['short_puts'] + [0] * (len(dates) - len(self.backtest_data['component_pnl']['short_puts']))
        })
        
        # Calculate cumulative P&L by component
        component_pnl_df['cum_long_calls'] = component_pnl_df['long_calls'].cumsum()
        component_pnl_df['cum_long_puts'] = component_pnl_df['long_puts'].cumsum()
        component_pnl_df['cum_short_calls'] = component_pnl_df['short_calls'].cumsum()
        component_pnl_df['cum_short_puts'] = component_pnl_df['short_puts'].cumsum()
        
        # Plot cumulative P&L by component
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_long_calls'], label='Long Calls', linewidth=1.5)
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_long_puts'], label='Long Puts', linewidth=1.5)
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_short_calls'], label='Short Calls', linewidth=1.5)
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_short_puts'], label='Short Puts', linewidth=1.5)
        
        # Calculate and plot total P&L
        component_pnl_df['cum_total'] = component_pnl_df['cum_long_calls'] + component_pnl_df['cum_long_puts'] + \
                                       component_pnl_df['cum_short_calls'] + component_pnl_df['cum_short_puts']
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_total'], label='Total P&L', color='black', linewidth=2)
        
        ax3.set_title('Cumulative P&L by Component')
        ax3.set_ylabel('P&L ($)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True)
        
        # Add a secondary plot showing component contribution
        ax4 = ax3.twinx()
        
        # Calculate daily component contribution as percentage
        for component in ['long_calls', 'long_puts', 'short_calls', 'short_puts']:
            component_pnl_df[f'{component}_pct'] = component_pnl_df[component][1:] / portfolio_values[:-1] * 100
            
        # Create a stacked area chart
        ax4.stackplot(component_pnl_df['date'], 
                      component_pnl_df['long_calls_pct'],
                      component_pnl_df['long_puts_pct'],
                      component_pnl_df['short_calls_pct'],
                      component_pnl_df['short_puts_pct'],
                      labels=['Long Calls %', 'Long Puts %', 'Short Calls %', 'Short Puts %'],
                      alpha=0.3)
                      
        ax4.set_ylabel('Daily P&L Contribution (%)')
        ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        plt.tight_layout()
        
        # Save figure
        plt_path = os.path.join(self.dir_, f'option_performance.{self.save_format}')
        plt.savefig(plt_path)
        
        # Create additional figure for component P&L analysis
        fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot daily P&L by component as bar chart
        width = 0.2
        x = np.arange(len(dates))
        
        ax5.bar(x - 1.5*width, component_pnl_df['long_calls'], width, label='Long Calls')
        ax5.bar(x - 0.5*width, component_pnl_df['long_puts'], width, label='Long Puts')
        ax5.bar(x + 0.5*width, component_pnl_df['short_calls'], width, label='Short Calls')
        ax5.bar(x + 1.5*width, component_pnl_df['short_puts'], width, label='Short Puts')
        
        # Set x-axis labels to dates
        ax5.set_xticks(x)
        ax5.set_xticklabels([d.strftime('%Y-%m-%d') if i % 10 == 0 else '' for i, d in enumerate(dates)], rotation=45)
        
        ax5.set_title('Daily P&L by Component')
        ax5.set_ylabel('P&L ($)')
        ax5.legend()
        ax5.grid(True)
        
        # Calculate and plot component P&L distribution
        component_names = ['Long Calls', 'Long Puts', 'Short Calls', 'Short Puts']
        component_data = [
            component_pnl_df['long_calls'],
            component_pnl_df['long_puts'],
            component_pnl_df['short_calls'],
            component_pnl_df['short_puts']
        ]
        
        ax6.boxplot(component_data, labels=component_names, patch_artist=True)
        
        # Add statistics for each component
        stats_text = ""
        for i, component in enumerate(['long_calls', 'long_puts', 'short_calls', 'short_puts']):
            data = component_pnl_df[component].dropna()
            if len(data) > 0:
                total = data.sum()
                mean = data.mean()
                std = data.std()
                sharpe = mean / std if std > 0 else 0
                win_rate = (data > 0).mean() * 100
                
                stats_text += f"{component_names[i]}: Sum=${total:.2f}, Mean=${mean:.2f}, Std=${std:.2f}, Sharpe={sharpe:.2f}, Win%={win_rate:.1f}%\n"
        
        # Add text with statistics
        ax6.text(0.02, 0.95, stats_text, transform=ax6.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax6.set_title('P&L Distribution by Component')
        ax6.set_ylabel('P&L ($)')
        ax6.grid(True)
        
        plt.tight_layout()
        
        # Save component analysis figure
        plt_path2 = os.path.join(self.dir_, f'component_pnl_analysis.{self.save_format}')
        plt.savefig(plt_path2)
        
        return fig
        '''
    def plot_performance(self, figsize=(18, 30)):
        """
        Enhanced plot_performance method with six subplots in single column layout
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import matplotlib.dates as mdates
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Set a nice color palette
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Create a figure with 6 subplots, each in its own row
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=figsize)
        
        # Convert dates to proper format
        dates = pd.to_datetime(self.trading_dates[:len(self.backtest_data['portfolio_values'])])
        
        # Plot portfolio values on ax1
        portfolio_values = np.array(self.backtest_data['portfolio_values'])
        initial_value = portfolio_values[0]
        
        # Calculate relative returns for better visualization
        relative_portfolio_values = portfolio_values / portfolio_values[0]
        
        portfolio_line, = ax1.plot(dates, relative_portfolio_values, label='Option Strategy', 
                                linewidth=2, color=colors[0])
        
        # Add benchmark if available
        if 'benchmark_values' in self.backtest_data and len(self.backtest_data['benchmark_values']) > 1:
            # Rescale benchmark to same starting value for comparison
            benchmark_values = np.array(self.backtest_data['benchmark_values'])
            relative_benchmark = benchmark_values / benchmark_values[0]
            
            benchmark_line, = ax1.plot(
                dates[:len(relative_benchmark)],
                relative_benchmark,
                label='Benchmark',
                linestyle='--',
                color=colors[1]
            )
            
            # Add legend with performance metrics
            if hasattr(self, 'performance_metrics'):
                portfolio_label = f'Option Strategy (Ann. Return: {self.performance_metrics["annualized_return"]:.2f}%, Sharpe: {self.performance_metrics["sharpe_ratio"]:.2f})'
                benchmark_ann_return = (benchmark_values[-1] / benchmark_values[0]) ** (252 / len(benchmark_values)) - 1
                benchmark_label = f'Benchmark (Ann. Return: {benchmark_ann_return*100:.2f}%)'
                
                portfolio_line.set_label(portfolio_label)
                benchmark_line.set_label(benchmark_label)
        
        ax1.set_title('Option Strategy Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Relative Value (1.0 = Initial)')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot daily returns on ax2
        daily_returns = np.array(self.backtest_data['daily_returns'])
        ax2.bar(dates, daily_returns * 100, color=colors[2], alpha=0.7)  # Convert to percentage
        
        ax2.set_title('Daily Returns', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Daily Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Create component P&L DataFrame
        component_pnl_df = pd.DataFrame({
            'date': dates,
            'long_calls': self.backtest_data['component_pnl']['long_calls'] + [0] * (len(dates) - len(self.backtest_data['component_pnl']['long_calls'])),
            'long_puts': self.backtest_data['component_pnl']['long_puts'] + [0] * (len(dates) - len(self.backtest_data['component_pnl']['long_puts'])),
            'short_calls': self.backtest_data['component_pnl']['short_calls'] + [0] * (len(dates) - len(self.backtest_data['component_pnl']['short_calls'])),
            'short_puts': self.backtest_data['component_pnl']['short_puts'] + [0] * (len(dates) - len(self.backtest_data['component_pnl']['short_puts']))
        })

        # Calculate cumulative P&L by component
        component_pnl_df['cum_long_calls'] = component_pnl_df['long_calls'].cumsum()
        component_pnl_df['cum_long_puts'] = component_pnl_df['long_puts'].cumsum()
        component_pnl_df['cum_short_calls'] = component_pnl_df['short_calls'].cumsum()
        component_pnl_df['cum_short_puts'] = component_pnl_df['short_puts'].cumsum()

        # Convert cumulative P&L to percentage of initial portfolio value
        component_pnl_df['cum_long_calls_pct'] = component_pnl_df['cum_long_calls'] / initial_value * 100
        component_pnl_df['cum_long_puts_pct'] = component_pnl_df['cum_long_puts'] / initial_value * 100
        component_pnl_df['cum_short_calls_pct'] = component_pnl_df['cum_short_calls'] / initial_value * 100
        component_pnl_df['cum_short_puts_pct'] = component_pnl_df['cum_short_puts'] / initial_value * 100

        # Calculate total cumulative P&L as percentage
        component_pnl_df['cum_total_pct'] = (
            component_pnl_df['cum_long_calls_pct'] + 
            component_pnl_df['cum_long_puts_pct'] + 
            component_pnl_df['cum_short_calls_pct'] + 
            component_pnl_df['cum_short_puts_pct']
        )

        # Plot cumulative P&L by component as percentages on ax3
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_long_calls_pct'], 
                label='Long Calls', linewidth=2, color=colors[0])
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_long_puts_pct'], 
                label='Long Puts', linewidth=2, color=colors[1])
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_short_calls_pct'], 
                label='Short Calls', linewidth=2, color=colors[2])
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_short_puts_pct'], 
                label='Short Puts', linewidth=2, color=colors[3])
        ax3.plot(component_pnl_df['date'], component_pnl_df['cum_total_pct'], 
                label='Total P&L', color='black', linewidth=2.5)

        # Format y-axis as percentage
        ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax3.set_title('Cumulative P&L by Component (% of Initial Portfolio)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('P&L (%)')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Daily component contribution as percentage on ax4
        # Make sure portfolio_values and component data arrays have the same length
        if len(portfolio_values) > len(component_pnl_df):
            # If portfolio_values has one extra element (initial value), use the rest
            portfolio_values_subset = portfolio_values[1:]
        elif len(portfolio_values) < len(component_pnl_df):
            # If component_pnl_df is longer, trim it
            component_pnl_df = component_pnl_df.iloc[:len(portfolio_values)]
            portfolio_values_subset = portfolio_values
        else:
            # If they're already the same length, use as is
            portfolio_values_subset = portfolio_values

        # Calculate daily component contribution percentages
        for component in ['long_calls', 'long_puts', 'short_calls', 'short_puts']:
            if len(component_pnl_df[component]) == len(portfolio_values_subset):
                component_pnl_df[f'{component}_pct'] = component_pnl_df[component] / portfolio_values_subset * 100
            else:
                # If there's still a mismatch, create a series of zeros
                component_pnl_df[f'{component}_pct'] = np.zeros(len(component_pnl_df))

        # Create a stacked area chart with enhanced colors on ax4
        ax4.stackplot(component_pnl_df['date'], 
                    component_pnl_df['long_calls_pct'],
                    component_pnl_df['long_puts_pct'],
                    component_pnl_df['short_calls_pct'],
                    component_pnl_df['short_puts_pct'],
                    labels=['Long Calls', 'Long Puts', 'Short Calls', 'Short Puts'],
                    colors=colors[:4],
                    alpha=0.7)
                    
        ax4.set_title('Daily P&L Contribution by Component', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Daily P&L Contribution (%)')
        ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # P&L distribution boxplot on ax5
        component_data = [
            component_pnl_df['long_calls'],
            component_pnl_df['long_puts'],
            component_pnl_df['short_calls'],
            component_pnl_df['short_puts']
        ]
        
        # Customize boxplot appearance
        boxprops = dict(linestyle='-', linewidth=2)
        flierprops = dict(marker='o', markersize=5, markerfacecolor='gray')
        medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
        whiskerprops = dict(linestyle='--', linewidth=1.5)
        
        # Create the boxplot with custom colors on ax5
        bp = ax5.boxplot(component_data, patch_artist=True, 
                        boxprops=boxprops, flierprops=flierprops,
                        medianprops=medianprops, whiskerprops=whiskerprops)
        
        # Set box colors using the same color scheme as lines
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=colors[i], alpha=0.7)
        
        # Set axis labels and title
        ax5.set_title('P&L Distribution by Component', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Daily P&L ($)')
        ax5.set_xticklabels(['Long Calls', 'Long Puts', 'Short Calls', 'Short Puts'], 
                        fontsize=11, rotation=0)
        ax5.grid(True, axis='y', alpha=0.3)
        
        # Add statistics text on the plot
        stats_text = ""
        for i, component in enumerate(['long_calls', 'long_puts', 'short_calls', 'short_puts']):
            data = component_pnl_df[component].dropna()
            if len(data) > 0:
                component_name = ['Long Calls', 'Long Puts', 'Short Calls', 'Short Puts'][i]
                total = data.sum()
                total_pct = total / initial_value * 100
                mean = data.mean()
                std = data.std()
                sharpe = mean / std if std > 0 else 0
                win_rate = (data > 0).mean() * 100
                
                stats_text += f"{component_name}: Sum=${total:.2f} ({total_pct:.2f}%), Mean=${mean:.2f}, Std=${std:.2f}, Sharpe={sharpe:.2f}, Win%={win_rate:.1f}%\n"
        
        # Add text with statistics in a nice box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax5.text(0.02, 0.95, stats_text, transform=ax5.transAxes, verticalalignment='top',
                fontsize=10, bbox=props)
        
        # Create total P&L contribution bar chart on ax6
        # Calculate total P&L for each component
        total_pnl = {
            'Long Calls': component_pnl_df['long_calls'].sum(),
            'Long Puts': component_pnl_df['long_puts'].sum(),
            'Short Calls': component_pnl_df['short_calls'].sum(),
            'Short Puts': component_pnl_df['short_puts'].sum()
        }
        
        # Calculate P&L percentage contribution
        pnl_contribution = {k: (v/initial_value*100) for k, v in total_pnl.items()}
        
        # Create bar chart
        components = list(pnl_contribution.keys())
        values = list(pnl_contribution.values())
        
        # Use horizontal bar chart for better label visibility
        bar_colors = [colors[i] for i in range(4)]
        bars = ax6.barh(components, values, color=bar_colors, alpha=0.8)
        
        # Add data labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + 0.5 if width > 0 else width - 2.5
            ax6.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{values[i]:.2f}%', va='center', fontsize=11,
                    fontweight='bold', color='black')
        
        # Add a total bar
        total_contribution = sum(values)
        ax6.barh(['TOTAL'], [total_contribution], color='black', alpha=0.7)
        ax6.text(total_contribution + 0.5 if total_contribution > 0 else total_contribution - 2.5, 
                len(components), f'{total_contribution:.2f}%', va='center', fontsize=11,
                fontweight='bold', color='black')
        
        # Set title and labels
        ax6.set_title('Total P&L Contribution (% of Initial Capital)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('P&L Contribution (%)')
        
        # Format x-axis as percentage
        ax6.xaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Add gridlines and style
        ax6.grid(True, axis='x', alpha=0.3)
        for spine in ['top', 'right']:
            ax6.spines[spine].set_visible(False)
        
        # Adjust layout for better spacing
        plt.tight_layout()
        
        # Save figure with high resolution
        plt_path = os.path.join(self.dir_, f'option_performance.{self.save_format}')
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        
        return fig