import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OptionStrategyBase:
    """Base class for option strategy generation"""
    
    def __init__(self, factor_list=None):
        """
        Initialize strategy with required factors
        
        Parameters:
        -----------
        factor_list : list
            List of factors required by this strategy
        """
        self.factor_list = factor_list or []
        
    def get_factor_list(self):
        """Return list of factors required by this strategy"""
        return self.factor_list
    
    def generate_option_targets(self, date, data_df, db_conn, min_price=5.0, 
                               market_cap_percentile=0.5, position_count=10, logger=print):
        """
        Generate option targets based on strategy logic
        
        Parameters:
        -----------
        date : datetime or str
            Current rebalancing date
        data_df : DataFrame
            Daily stock data including factors
        db_conn : duckdb.DuckDBPyConnection
            Database connection for option data
        min_price : float
            Minimum price threshold for underlying stocks
        market_cap_percentile : float
            Market cap percentile threshold (0-1)
        position_count : int
            Number of option positions to generate
        logger : function
            Logging function
            
        Returns:
        --------
        dict : Option targets
        """
        raise NotImplementedError("Subclasses must implement generate_option_targets")

'''
class ROEOptionStrategy(OptionStrategyBase):
    """
    Option strategy that trades calls and puts based on underlying stock ROE:
    - Long calls and puts on high ROE stocks (straddles)
    - Short calls and puts on low ROE stocks (straddles)
    
    This version uses notional-based position sizing based on initial cash.
    """
    
    def __init__(self, top_pct=0.01, bottom_pct=0.01, days_to_expiry_range=(20, 40),
                target_delta_call=0.5, target_delta_put=-0.5, min_open_interest=10,
                weekly_trade_day=4, holding_period_days=7, store_intermediate=True,
                notional_pct=0.01, max_total_notional=0.5, initial_cash=1000000):
        """
        Initialize strategy
        
        Parameters:
        -----------
        top_pct : float
            Percentile for top ROE stocks (0.01 = top 1%)
        bottom_pct : float
            Percentile for bottom ROE stocks (0.01 = bottom 1%)
        days_to_expiry_range : tuple
            Target (min, max) days to expiration for options
        target_delta_call : float
            Target delta for call options (0.5 = ATM)
        target_delta_put : float
            Target delta for put options (-0.5 = ATM)
        min_open_interest : int
            Minimum open interest for option liquidity filter
        weekly_trade_day : int
            Day of week for trading (0=Monday, 4=Friday)
        holding_period_days : int
            Number of days to hold positions before closing
        store_intermediate : bool
            Whether to store intermediate results
        notional_pct : float
            Percentage of initial cash to allocate to each stock (0.01 = 1%)
        max_total_notional : float
            Maximum total notional as percentage of initial cash (0.5 = 50%)
        initial_cash : float
            Initial cash amount for sizing calculations
        """
        # Define the factors required by this strategy
        factor_list = [
            'roe',              # Primary ranking factor
        ]
        
        # Call parent constructor with factor list
        super().__init__(factor_list)
        
        # Store strategy parameters
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.days_to_expiry_range = days_to_expiry_range
        self.target_delta_call = target_delta_call
        self.target_delta_put = target_delta_put
        self.min_open_interest = min_open_interest
        self.weekly_trade_day = weekly_trade_day
        self.holding_period_days = holding_period_days
        self.store_intermediate = store_intermediate
        
        # Position sizing parameters
        self.notional_pct = notional_pct
        self.max_total_notional = max_total_notional
        self.initial_cash = initial_cash
        
        # Storage for intermediate results
        self.intermediate_results = {}
    
    def is_trading_day(self, date):
        """Check if this is a rebalancing day"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return date.weekday() == self.weekly_trade_day

    def generate_option_targets(self, date, data_df, db_path, min_price=5.0, 
                                market_cap_percentile=0.9, position_count=10, logger=print):
        """
        Generate option targets with individual calls and puts using notional-based sizing
        
        Parameters:
        -----------
        date : datetime or str
            Current date
        data_df : DataFrame
            Stock data with factors and option availability flags
        db_path : str
            Path to the database file
        min_price : float
            Minimum price for underlying stocks
        market_cap_percentile : float
            Market cap percentile threshold
        position_count : int
            Target number of option positions per type
        logger : function
            Logging function
            
        Returns:
        --------
        dict : Option targets
        """
        # Reset intermediate storage
        if self.store_intermediate:
            self.intermediate_results = {}
            
        # Format date if needed
        if isinstance(date, str):
            date_obj = pd.to_datetime(date)
            date_str = date
        else:
            date_obj = date
            date_str = date.strftime('%Y-%m-%d')
        
        # Check if this is a rebalancing day
        is_rebalance = self.is_trading_day(date_obj)
        
        # Initialize targets with separate option categories
        targets = {
            'long_calls': [],
            'long_puts': [],
            'short_calls': [],
            'short_puts': [],
            'rebalance': is_rebalance
        }
        
        # If not a rebalancing day, return empty targets
        if not is_rebalance:
            logger(f"Not a rebalancing day: {date_str}")
            return targets
        
        # Step 1: First ensure all required factors exist in the data
        for factor in self.factor_list:
            if factor not in data_df.columns:
                logger(f"Required factor '{factor}' not found in data")
                return targets  # Return empty targets if any factor is missing
        
        try:
            # Convert columns to numeric before filtering to avoid type issues
            if 'dlyprc' in data_df.columns and pd.api.types.is_object_dtype(data_df['dlyprc']):
                data_df['dlyprc'] = pd.to_numeric(data_df['dlyprc'], errors='coerce')
                
            if 'dlycap' in data_df.columns and pd.api.types.is_object_dtype(data_df['dlycap']):
                data_df['dlycap'] = pd.to_numeric(data_df['dlycap'], errors='coerce')
            
            # Initial conditions
            condition = (
                (data_df['dlyprc'] > min_price) &
                (data_df['dlycap'] > data_df['dlycap'].quantile(market_cap_percentile)) &
                (data_df['option_available'] == 1) &
                (data_df['has_calls'] == 1) &
                (data_df['has_puts'] == 1) &
                (data_df['secid'].notna()) &
                (data_df['secid'] > 0)
            )

            # Combine non-null checks for each factor
            for factor in self.factor_list:
                condition &= (~data_df[factor].isna())

            universe_df = data_df[condition].copy()
        
        except Exception as e:
            logger(f"Error filtering stock universe: {str(e)}")
            import traceback
            logger(traceback.format_exc())
            # Create empty universe to avoid further errors
            universe_df = data_df.iloc[0:0].copy()
        
        # Track intermediate result
        if self.store_intermediate:
            self.intermediate_results['initial_universe'] = universe_df.copy()
            
        logger(f"Starting universe: {len(universe_df)} stocks with valid options and factors")
        
        # Check if we have enough stocks
        if len(universe_df) < 2:
            logger("Not enough stocks with options and valid factors available")
            return targets
        
        
        # STAGE 1: Rank stocks by ROE and other factors
        logger("Stage 1: Ranking stocks by factors")
        
        # Check if required factors exist
        missing_factors = [f for f in self.factor_list if f not in universe_df.columns]
        if missing_factors:
            logger(f"Warning: Missing factors: {missing_factors}")
            
        # Only use available factors
        available_factors = [f for f in self.factor_list if f in universe_df.columns]
        if not available_factors:
            logger("Error: No required factors available")
            return targets
        
        # Rank stocks by each factor
        for factor in available_factors:
            rank_col = f"{factor}_rank"
            # Create absolute value column
            abs_col = f"{factor}_abs"
            universe_df[abs_col] = universe_df[factor].abs()
            universe_df[rank_col] = universe_df[abs_col].rank(ascending=True)
        
        # Calculate combined rank (equal weight for now)
        rank_columns = [f"{f}_rank" for f in available_factors]
        universe_df['combined_rank'] = universe_df[rank_columns].sum(axis=1)
        
        # Sort by combined rank
        universe_df.sort_values('combined_rank', ascending=False, inplace=True)
        
        # Track intermediate result
        if self.store_intermediate:
            self.intermediate_results['ranked_universe'] = universe_df.copy()
        
        # STAGE 2: Identify top and bottom stocks
        logger("Stage 2: Identifying top and bottom stocks")
        
        # Calculate percentile thresholds
        stock_count = len(universe_df)
        top_count = max(1, int(stock_count * self.top_pct))
        bottom_count = max(1, int(stock_count * self.bottom_pct))
        
        # Select top and bottom stocks
        top_stocks = universe_df.iloc[:top_count]
        bottom_stocks = universe_df.iloc[-bottom_count:]
        
        logger(f"Selected {len(top_stocks)} high ROE stocks and {len(bottom_stocks)} low ROE stocks")
        
        # Track intermediate results
        if self.store_intermediate:
            self.intermediate_results['top_stocks'] = top_stocks.copy()
            self.intermediate_results['bottom_stocks'] = bottom_stocks.copy()
        
        # STAGE 3: Find options for selected stocks
        logger("Stage 3: Finding options")
        
        # Import required functions
        from option_tools import get_option_data_for_stock_fromdb, select_option_by_delta
        
        # Parameters for option filtering
        delta_ranges = {
            'C': (self.target_delta_call - 0.05, self.target_delta_call + 0.05),
            'P': (self.target_delta_put - 0.05, self.target_delta_put + 0.05)
        }
        
        # How many positions to try to build (limit by position_count)
        long_count = min(position_count, len(top_stocks))
        short_count = min(position_count, len(bottom_stocks))
        
        # Calculate the notional amount per stock
        notional_per_stock = self.initial_cash * self.notional_pct
        
        # Split the notional amount between call and put for each stock
        notional_per_option = notional_per_stock / 2
        
        # Calculate maximum number of stocks based on max total notional
        max_stocks = int(self.max_total_notional / self.notional_pct)
        
        # Limit position counts based on max stocks
        long_count = min(long_count, max_stocks // 2)  # Half for long, half for short
        short_count = min(short_count, max_stocks // 2)
        
        logger(f"Position sizing: ${notional_per_stock:,.2f} per stock, ${notional_per_option:,.2f} per option")
        logger(f"Maximum positions: {max_stocks} stocks (based on {self.max_total_notional:.1%} max notional)")
        
        # Calculate target date to close positions
        target_close_date = date_obj + pd.Timedelta(days=self.holding_period_days)
        target_close_date_str = target_close_date.strftime('%Y-%m-%d')
        
        # Process top stocks for long options
        long_calls = []
        long_puts = []
        processed_long = 0
        total_long_notional = 0
        
        for _, row in top_stocks.iterrows():
            if processed_long >= long_count:
                break
                
            permno = row.name
            secid = row.get('secid')
            
            # Skip stocks without valid secid
            if pd.isna(secid) or secid <= 0:
                continue
                
            try:
                # Get options for this stock with db_path
                options = get_option_data_for_stock_fromdb(
                    date_str=date_str,
                    permno=permno,
                    db_path=db_path,  # Pass the database path
                    days_to_expiry_range=self.days_to_expiry_range,
                    delta_ranges=delta_ranges,
                    min_open_interest=self.min_open_interest
                )
                
                if len(options) == 0:
                    continue
                    
                # Find best call and put separately
                call_options = options[options['cp_flag'] == 'C'].copy()
                put_options = options[options['cp_flag'] == 'P'].copy()
                
                if len(call_options) > 0 and len(put_options) > 0:
                    # Get best call by delta
                    best_call = select_option_by_delta(
                        call_options,
                        target_delta=self.target_delta_call,
                        target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                        option_type='C'
                    )
                    
                    # Get best put by delta
                    best_put = select_option_by_delta(
                        put_options,
                        target_delta=self.target_delta_put,
                        target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                        option_type='P'
                    )
                    
                    # Add call to targets if found - using notional instead of weight
                    if best_call is not None:
                        # Instead of weight, store notional amount
                        long_calls.append({
                            'optionid': best_call['optionid'],
                            'permno': permno,
                            'strike': best_call['strike_price'],
                            'expiry': best_call['exdate'],
                            'notional': notional_per_option,  # Fixed notional amount
                            'target_date': target_close_date_str,
                            'strategy': 'long_straddle_call',
                            'delta': best_call['delta'],
                            # Include bid-ask for price reference
                            'bidprice': best_call.get('bidprice', 0),
                            'askprice': best_call.get('askprice', 0),
                            'midprice': best_call.get('midprice', (best_call.get('bidprice', 0) + best_call.get('askprice', 0)) / 2)
                        })
                        total_long_notional += notional_per_option
                    
                    # Add put to targets if found - using notional instead of weight
                    if best_put is not None:
                        long_puts.append({
                            'optionid': best_put['optionid'],
                            'permno': permno,
                            'strike': best_put['strike_price'],
                            'expiry': best_put['exdate'],
                            'notional': notional_per_option,  # Fixed notional amount
                            'target_date': target_close_date_str,
                            'strategy': 'long_straddle_put',
                            'delta': best_put['delta'],
                            # Include bid-ask for price reference
                            'bidprice': best_put.get('bidprice', 0),
                            'askprice': best_put.get('askprice', 0),
                            'midprice': best_put.get('midprice', (best_put.get('bidprice', 0) + best_put.get('askprice', 0)) / 2)
                        })
                        total_long_notional += notional_per_option
                    
                    # Count as processed if at least one option was added
                    if best_call is not None or best_put is not None:
                        processed_long += 1
            except Exception as e:
                logger(f"Error processing options for permno {permno}: {str(e)}")
                continue
        
        # Process bottom stocks for short options
        short_calls = []
        short_puts = []
        processed_short = 0
        total_short_notional = 0
        
        for _, row in bottom_stocks.iterrows():
            if processed_short >= short_count:
                break
                
            permno = row.name
            secid = row.get('secid')
            
            # Skip stocks without valid secid
            if pd.isna(secid) or secid <= 0:
                continue
                
            try:
                # Get options for this stock with db_path
                options = get_option_data_for_stock_fromdb(
                    date_str=date_str,
                    permno=permno,
                    db_path=db_path,  # Pass the database path
                    days_to_expiry_range=self.days_to_expiry_range,
                    delta_ranges=delta_ranges,
                    min_open_interest=self.min_open_interest
                )
                
                if len(options) == 0:
                    continue
                    
                # Find best call and put separately
                call_options = options[options['cp_flag'] == 'C'].copy()
                put_options = options[options['cp_flag'] == 'P'].copy()
                
                if len(call_options) > 0 and len(put_options) > 0:
                    # Get best call by delta
                    best_call = select_option_by_delta(
                        call_options,
                        target_delta=self.target_delta_call,
                        target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                        option_type='C'
                    )
                    
                    # Get best put by delta
                    best_put = select_option_by_delta(
                        put_options,
                        target_delta=self.target_delta_put,
                        target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                        option_type='P'
                    )
                    
                    # For shorts, use a smaller notional to manage risk
                    short_sizing_factor = 0.8  # 80% of the long notional
                    short_notional = notional_per_option * short_sizing_factor
                    
                    # Add call to targets if found - using notional instead of weight
                    if best_call is not None:
                        short_calls.append({
                            'optionid': best_call['optionid'],
                            'permno': permno,
                            'strike': best_call['strike_price'],
                            'expiry': best_call['exdate'],
                            'notional': short_notional,  # Fixed notional with risk adjustment
                            'target_date': target_close_date_str,
                            'strategy': 'short_straddle_call',
                            'delta': best_call['delta'],
                            # Include bid-ask for price reference
                            'bidprice': best_call.get('bidprice', 0),
                            'askprice': best_call.get('askprice', 0),
                            'midprice': best_call.get('midprice', (best_call.get('bidprice', 0) + best_call.get('askprice', 0)) / 2)
                        })
                        total_short_notional += short_notional
                    
                    # Add put to targets if found - using notional instead of weight
                    if best_put is not None:
                        short_puts.append({
                            'optionid': best_put['optionid'],
                            'permno': permno,
                            'strike': best_put['strike_price'],
                            'expiry': best_put['exdate'],
                            'notional': short_notional,  # Fixed notional with risk adjustment
                            'target_date': target_close_date_str,
                            'strategy': 'short_straddle_put',
                            'delta': best_put['delta'],
                            # Include bid-ask for price reference
                            'bidprice': best_put.get('bidprice', 0),
                            'askprice': best_put.get('askprice', 0),
                            'midprice': best_put.get('midprice', (best_put.get('bidprice', 0) + best_put.get('askprice', 0)) / 2)
                        })
                        total_short_notional += short_notional
                    
                    # Count as processed if at least one option was added
                    if best_call is not None or best_put is not None:
                        processed_short += 1
            except Exception as e:
                logger(f"Error processing options for permno {permno}: {str(e)}")
                continue
        
        # Add options to targets
        targets['long_calls'] = long_calls
        targets['long_puts'] = long_puts
        targets['short_calls'] = short_calls
        targets['short_puts'] = short_puts
        
        # Log summary
        total_notional = total_long_notional + total_short_notional
        total_notional_pct = total_notional / self.initial_cash
        
        logger(f"Final option targets:")
        logger(f"- Long calls: {len(long_calls)}")
        logger(f"- Long puts: {len(long_puts)}")
        logger(f"- Short calls: {len(short_calls)}")
        logger(f"- Short puts: {len(short_puts)}")
        logger(f"Total notional: ${total_notional:,.2f} ({total_notional_pct:.2%} of initial cash)")
        logger(f"All positions will be held until {target_close_date_str}")
        
        self.save_intermediate_results(date_obj)
        return targets

    def get_intermediate_results(self):
        """Get stored intermediate results"""
        if not self.store_intermediate:
            return {"error": "Intermediate result storage is not enabled"}
        return self.intermediate_results
    
    def save_intermediate_results(self, date, directory="option_intermediate_results"):
        """Save intermediate results to disk"""
        import os
        
        if not self.store_intermediate:
            return
            
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        date_str = date.strftime('%Y%m%d')
        
        # Save each DataFrame
        for stage, df in self.intermediate_results.items():
            if isinstance(df, pd.DataFrame):
                filename = f"{directory}/{date_str}_{stage}.csv"
                df.to_csv(filename)

'''

class ROEOptionStrategy(OptionStrategyBase):
    """
    Option strategy that trades calls and puts based on underlying stock ROE:
    - Long ATM calls and puts on high ROE stocks (straddles)
    - Short 25-delta calls and puts on low ROE stocks (OTM straddles)
    
    This version uses notional-based position sizing based on initial cash.
    """
    
    def __init__(self, top_pct=0.01, bottom_pct=0.01, days_to_expiry_range=(20, 40),
                target_delta_call=0.5, target_delta_put=-0.5, 
                target_delta_short_call=0.25, target_delta_short_put=-0.25,
                min_open_interest=10, weekly_trade_day=4, holding_period_days=7, 
                store_intermediate=True, notional_pct=0.01, max_total_notional=0.5, 
                initial_cash=1000000):
        """
        Initialize strategy
        
        Parameters:
        -----------
        top_pct : float
            Percentile for top ROE stocks (0.01 = top 1%)
        bottom_pct : float
            Percentile for bottom ROE stocks (0.01 = bottom 1%)
        days_to_expiry_range : tuple
            Target (min, max) days to expiration for options
        target_delta_call : float
            Target delta for long call options (0.5 = ATM)
        target_delta_put : float
            Target delta for long put options (-0.5 = ATM)
        target_delta_short_call : float
            Target delta for short call options (0.25 = OTM)
        target_delta_short_put : float
            Target delta for short put options (-0.25 = OTM)
        min_open_interest : int
            Minimum open interest for option liquidity filter
        weekly_trade_day : int
            Day of week for trading (0=Monday, 4=Friday)
        holding_period_days : int
            Number of days to hold positions before closing
        store_intermediate : bool
            Whether to store intermediate results
        notional_pct : float
            Percentage of initial cash to allocate to each stock (0.01 = 1%)
        max_total_notional : float
            Maximum total notional as percentage of initial cash (0.5 = 50%)
        initial_cash : float
            Initial cash amount for sizing calculations
        """
        # Define the factors required by this strategy
        factor_list = [
            'cash_based_op_profitability',              # Primary ranking factor
        ]
        
        # Call parent constructor with factor list
        super().__init__(factor_list)
        
        # Store strategy parameters
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.days_to_expiry_range = days_to_expiry_range
        self.target_delta_call = target_delta_call
        self.target_delta_put = target_delta_put
        
        # New target deltas for short positions
        self.target_delta_short_call = target_delta_short_call
        self.target_delta_short_put = target_delta_short_put
        
        self.min_open_interest = min_open_interest
        self.weekly_trade_day = weekly_trade_day
        self.holding_period_days = holding_period_days
        self.store_intermediate = store_intermediate
        
        # Position sizing parameters
        self.notional_pct = notional_pct
        self.max_total_notional = max_total_notional
        self.initial_cash = initial_cash
        
        # Storage for intermediate results
        self.intermediate_results = {}
    
    def is_trading_day(self, date):
        """Check if this is a rebalancing day"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return date.weekday() == self.weekly_trade_day

    def generate_option_targets(self, date, data_df, db_path, min_price=5.0, 
                                market_cap_percentile=0.9, position_count=10, logger=print):
        """
        Generate option targets with individual calls and puts using notional-based sizing
        
        Parameters:
        -----------
        date : datetime or str
            Current date
        data_df : DataFrame
            Stock data with factors and option availability flags
        db_path : str
            Path to the database file
        min_price : float
            Minimum price for underlying stocks
        market_cap_percentile : float
            Market cap percentile threshold
        position_count : int
            Target number of option positions per type
        logger : function
            Logging function
            
        Returns:
        --------
        dict : Option targets
        """
        # Reset intermediate storage
        if self.store_intermediate:
            self.intermediate_results = {}
            
        # Format date if needed
        if isinstance(date, str):
            date_obj = pd.to_datetime(date)
            date_str = date
        else:
            date_obj = date
            date_str = date.strftime('%Y-%m-%d')
        
        # Check if this is a rebalancing day
        is_rebalance = self.is_trading_day(date_obj)
        
        # Initialize targets with separate option categories
        targets = {
            'long_calls': [],
            'long_puts': [],
            'short_calls': [],
            'short_puts': [],
            'rebalance': is_rebalance
        }
        
        # If not a rebalancing day, return empty targets
        if not is_rebalance:
            logger(f"Not a rebalancing day: {date_str}")
            return targets
        
        # Step 1: First ensure all required factors exist in the data
        for factor in self.factor_list:
            if factor not in data_df.columns:
                logger(f"Required factor '{factor}' not found in data")
                return targets  # Return empty targets if any factor is missing
        
        try:
            # Convert columns to numeric before filtering to avoid type issues
            if 'dlyprc' in data_df.columns and pd.api.types.is_object_dtype(data_df['dlyprc']):
                data_df['dlyprc'] = pd.to_numeric(data_df['dlyprc'], errors='coerce')
                
            if 'dlycap' in data_df.columns and pd.api.types.is_object_dtype(data_df['dlycap']):
                data_df['dlycap'] = pd.to_numeric(data_df['dlycap'], errors='coerce')
            
            # Initial conditions
            condition = (
                (data_df['dlyprc'] > min_price) &
                (data_df['dlycap'] > data_df['dlycap'].quantile(market_cap_percentile)) &
                (data_df['option_available'] == 1) &
                (data_df['has_calls'] == 1) &
                (data_df['has_puts'] == 1) &
                (data_df['secid'].notna()) &
                (data_df['secid'] > 0)
            )

            # Combine non-null checks for each factor
            for factor in self.factor_list:
                condition &= (~data_df[factor].isna())

            universe_df = data_df[condition].copy()
        
        except Exception as e:
            logger(f"Error filtering stock universe: {str(e)}")
            import traceback
            logger(traceback.format_exc())
            # Create empty universe to avoid further errors
            universe_df = data_df.iloc[0:0].copy()
        
        # Track intermediate result
        if self.store_intermediate:
            self.intermediate_results['initial_universe'] = universe_df.copy()
            
        logger(f"Starting universe: {len(universe_df)} stocks with valid options and factors")
        
        # Check if we have enough stocks
        if len(universe_df) < 2:
            logger("Not enough stocks with options and valid factors available")
            return targets
        
        
        # STAGE 1: Rank stocks by ROE and other factors
        logger("Stage 1: Ranking stocks by factors")
        
        # Check if required factors exist
        missing_factors = [f for f in self.factor_list if f not in universe_df.columns]
        if missing_factors:
            logger(f"Warning: Missing factors: {missing_factors}")
            
        # Only use available factors
        available_factors = [f for f in self.factor_list if f in universe_df.columns]
        if not available_factors:
            logger("Error: No required factors available")
            return targets
        
        # Rank stocks by each factor
        for factor in available_factors:
            rank_col = f"{factor}_rank"
            # Create absolute value column
            #abs_col = f"{factor}_abs"
            #universe_df[abs_col] = universe_df[factor].abs()
            universe_df[rank_col] = universe_df[factor].rank(ascending=True)
        
        # Calculate combined rank (equal weight for now)
        rank_columns = [f"{f}_rank" for f in available_factors]
        universe_df['combined_rank'] = universe_df[rank_columns].sum(axis=1)
        
        # Sort by combined rank
        universe_df.sort_values('combined_rank', ascending=False, inplace=True)
        
        # Track intermediate result
        if self.store_intermediate:
            self.intermediate_results['ranked_universe'] = universe_df.copy()
        
        # STAGE 2: Identify top and bottom stocks
        logger("Stage 2: Identifying top and bottom stocks")
        
        # Calculate percentile thresholds
        stock_count = len(universe_df)
        top_count = max(1, int(stock_count * self.top_pct))
        bottom_count = max(1, int(stock_count * self.bottom_pct))
        
        # Select top and bottom stocks
        top_stocks = universe_df.iloc[:top_count]
        bottom_stocks = universe_df.iloc[-bottom_count:]
        
        logger(f"Selected {len(top_stocks)} high ROE stocks and {len(bottom_stocks)} low ROE stocks")
        
        # Track intermediate results
        if self.store_intermediate:
            self.intermediate_results['top_stocks'] = top_stocks.copy()
            self.intermediate_results['bottom_stocks'] = bottom_stocks.copy()
        
        # STAGE 3: Find options for selected stocks
        logger("Stage 3: Finding options")
        
        # Import required functions
        from option_tools import get_option_data_for_stock_fromdb, select_option_by_delta
        
        # Parameters for long option filtering - ATM options (0.5 delta)
        long_delta_ranges = {
            'C': (self.target_delta_call - 0.05, self.target_delta_call + 0.05),
            'P': (self.target_delta_put - 0.05, self.target_delta_put + 0.05)
        }
        
        # Parameters for short option filtering - 25 delta OTM options
        short_delta_ranges = {
            'C': (self.target_delta_short_call - 0.05, self.target_delta_short_call + 0.05),
            'P': (self.target_delta_short_put - 0.05, self.target_delta_short_put + 0.05)
        }
        
        # How many positions to try to build (limit by position_count)
        long_count = min(position_count, len(top_stocks))
        short_count = min(position_count, len(bottom_stocks))
        
        # Calculate the notional amount per stock
        notional_per_stock = self.initial_cash * self.notional_pct
        
        # Split the notional amount between call and put for each stock
        notional_per_option = notional_per_stock / 2
        
        # Calculate maximum number of stocks based on max total notional
        max_stocks = int(self.max_total_notional / self.notional_pct)
        
        # Limit position counts based on max stocks
        long_count = min(long_count, max_stocks // 2)  # Half for long, half for short
        short_count = min(short_count, max_stocks // 2)
        
        logger(f"Position sizing: ${notional_per_stock:,.2f} per stock, ${notional_per_option:,.2f} per option")
        logger(f"Maximum positions: {max_stocks} stocks (based on {self.max_total_notional:.1%} max notional)")
        
        # Calculate target date to close positions
        target_close_date = date_obj + pd.Timedelta(days=self.holding_period_days)
        target_close_date_str = target_close_date.strftime('%Y-%m-%d')
        
        # Process top stocks for long options (ATM options)
        long_calls = []
        long_puts = []
        processed_long = 0
        total_long_notional = 0
        
        for _, row in top_stocks.iterrows():
            if processed_long >= long_count:
                break
                
            permno = row.name
            secid = row.get('secid')
            
            # Skip stocks without valid secid
            if pd.isna(secid) or secid <= 0:
                continue
                
            try:
                # Get options for this stock with db_path - using ATM delta ranges for longs
                options = get_option_data_for_stock_fromdb(
                    date_str=date_str,
                    permno=permno,
                    db_path=db_path,
                    days_to_expiry_range=self.days_to_expiry_range,
                    delta_ranges=long_delta_ranges,
                    min_open_interest=self.min_open_interest
                )
                
                if len(options) == 0:
                    continue
                    
                # Find best call and put separately
                call_options = options[options['cp_flag'] == 'C'].copy()
                put_options = options[options['cp_flag'] == 'P'].copy()
                
                if len(call_options) > 0 and len(put_options) > 0:
                    # Get best call by delta - targeting ATM (0.5 delta)
                    best_call = select_option_by_delta(
                        call_options,
                        target_delta=self.target_delta_call,
                        target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                        option_type='C'
                    )
                    
                    # Get best put by delta - targeting ATM (-0.5 delta)
                    best_put = select_option_by_delta(
                        put_options,
                        target_delta=self.target_delta_put,
                        target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                        option_type='P'
                    )
                    
                    # Add call to targets if found - using notional instead of weight
                    if best_call is not None:
                        # Instead of weight, store notional amount
                        long_calls.append({
                            'optionid': best_call['optionid'],
                            'permno': permno,
                            'strike': best_call['strike_price'],
                            'expiry': best_call['exdate'],
                            'notional': notional_per_option,  # Fixed notional amount
                            'target_date': target_close_date_str,
                            'strategy': 'long_straddle_call',
                            'delta': best_call['delta'],
                            # Include bid-ask for price reference
                            'bidprice': best_call.get('bidprice', 0),
                            'askprice': best_call.get('askprice', 0),
                            'midprice': best_call.get('midprice', (best_call.get('bidprice', 0) + best_call.get('askprice', 0)) / 2)
                        })
                        total_long_notional += notional_per_option
                    
                    # Add put to targets if found - using notional instead of weight
                    if best_put is not None:
                        long_puts.append({
                            'optionid': best_put['optionid'],
                            'permno': permno,
                            'strike': best_put['strike_price'],
                            'expiry': best_put['exdate'],
                            'notional': notional_per_option,  # Fixed notional amount
                            'target_date': target_close_date_str,
                            'strategy': 'long_straddle_put',
                            'delta': best_put['delta'],
                            # Include bid-ask for price reference
                            'bidprice': best_put.get('bidprice', 0),
                            'askprice': best_put.get('askprice', 0),
                            'midprice': best_put.get('midprice', (best_put.get('bidprice', 0) + best_put.get('askprice', 0)) / 2)
                        })
                        total_long_notional += notional_per_option
                    
                    # Count as processed if at least one option was added
                    if best_call is not None or best_put is not None:
                        processed_long += 1
            except Exception as e:
                logger(f"Error processing options for permno {permno}: {str(e)}")
                continue
        
        # Process bottom stocks for short options (25 delta OTM options)
        short_calls = []
        short_puts = []
        processed_short = 0
        total_short_notional = 0
        
        for _, row in bottom_stocks.iterrows():
            if processed_short >= short_count:
                break
                
            permno = row.name
            secid = row.get('secid')
            
            # Skip stocks without valid secid
            if pd.isna(secid) or secid <= 0:
                continue
                
            try:
                # Get options for this stock with db_path - using 25 delta ranges for shorts
                options = get_option_data_for_stock_fromdb(
                    date_str=date_str,
                    permno=permno,
                    db_path=db_path,
                    days_to_expiry_range=self.days_to_expiry_range,
                    delta_ranges=short_delta_ranges,  # Using OTM delta ranges for shorts
                    min_open_interest=self.min_open_interest
                )
                
                if len(options) == 0:
                    continue
                    
                # Find best call and put separately
                call_options = options[options['cp_flag'] == 'C'].copy()
                put_options = options[options['cp_flag'] == 'P'].copy()
                
                if len(call_options) > 0 and len(put_options) > 0:
                    # Get best call by delta - targeting 0.25 delta (OTM)
                    best_call = select_option_by_delta(
                        call_options,
                        target_delta=self.target_delta_short_call,  # 0.25 delta
                        target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                        option_type='C'
                    )
                    
                    # Get best put by delta - targeting -0.25 delta (OTM)
                    best_put = select_option_by_delta(
                        put_options,
                        target_delta=self.target_delta_short_put,  # -0.25 delta
                        target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                        option_type='P'
                    )
                    
                    # For shorts, use a smaller notional to manage risk
                    short_sizing_factor = 0.8  # 80% of the long notional
                    short_notional = notional_per_option * short_sizing_factor
                    
                    # Add call to targets if found - using notional instead of weight
                    if best_call is not None:
                        short_calls.append({
                            'optionid': best_call['optionid'],
                            'permno': permno,
                            'strike': best_call['strike_price'],
                            'expiry': best_call['exdate'],
                            'notional': short_notional,  # Fixed notional with risk adjustment
                            'target_date': target_close_date_str,
                            'strategy': 'short_straddle_call',
                            'delta': best_call['delta'],
                            # Include bid-ask for price reference
                            'bidprice': best_call.get('bidprice', 0),
                            'askprice': best_call.get('askprice', 0),
                            'midprice': best_call.get('midprice', (best_call.get('bidprice', 0) + best_call.get('askprice', 0)) / 2)
                        })
                        total_short_notional += short_notional
                    
                    # Add put to targets if found - using notional instead of weight
                    if best_put is not None:
                        short_puts.append({
                            'optionid': best_put['optionid'],
                            'permno': permno,
                            'strike': best_put['strike_price'],
                            'expiry': best_put['exdate'],
                            'notional': short_notional,  # Fixed notional with risk adjustment
                            'target_date': target_close_date_str,
                            'strategy': 'short_straddle_put',
                            'delta': best_put['delta'],
                            # Include bid-ask for price reference
                            'bidprice': best_put.get('bidprice', 0),
                            'askprice': best_put.get('askprice', 0),
                            'midprice': best_put.get('midprice', (best_put.get('bidprice', 0) + best_put.get('askprice', 0)) / 2)
                        })
                        total_short_notional += short_notional
                    
                    # Count as processed if at least one option was added
                    if best_call is not None or best_put is not None:
                        processed_short += 1
            except Exception as e:
                logger(f"Error processing options for permno {permno}: {str(e)}")
                continue
        
        # Add options to targets
        targets['long_calls'] = long_calls
        targets['long_puts'] = long_puts
        targets['short_calls'] = short_calls
        targets['short_puts'] = short_puts
        
        # Log summary
        total_notional = total_long_notional + total_short_notional
        total_notional_pct = total_notional / self.initial_cash
        
        logger(f"Final option targets:")
        logger(f"- Long calls (ATM, delta ≈ {self.target_delta_call}): {len(long_calls)}")
        logger(f"- Long puts (ATM, delta ≈ {self.target_delta_put}): {len(long_puts)}")
        logger(f"- Short calls (OTM, delta ≈ {self.target_delta_short_call}): {len(short_calls)}")
        logger(f"- Short puts (OTM, delta ≈ {self.target_delta_short_put}): {len(short_puts)}")
        logger(f"Total notional: ${total_notional:,.2f} ({total_notional_pct:.2%} of initial cash)")
        logger(f"All positions will be held until {target_close_date_str}")
        
        self.save_intermediate_results(date_obj)
        return targets

    def get_intermediate_results(self):
        """Get stored intermediate results"""
        if not self.store_intermediate:
            return {"error": "Intermediate result storage is not enabled"}
        return self.intermediate_results
    
    def save_intermediate_results(self, date, directory="option_intermediate_results"):
        """Save intermediate results to disk"""
        import os
        
        if not self.store_intermediate:
            return
            
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        date_str = date.strftime('%Y%m%d')
        
        # Save each DataFrame
        for stage, df in self.intermediate_results.items():
            if isinstance(df, pd.DataFrame):
                filename = f"{directory}/{date_str}_{stage}.csv"
                df.to_csv(filename)


class PutWriteStrategy(OptionStrategyBase):
    """
    Option strategy that writes puts on high-quality stocks
    with strong momentum, targeting specific delta levels
    """
    
    def __init__(self, top_pct=0.2, days_to_expiry_range=(20, 40),
                target_delta=-0.3, min_open_interest=20,
                weekly_trade_day=4, store_intermediate=True):
        """
        Initialize strategy
        
        Parameters:
        -----------
        top_pct : float
            Percentile for top stocks (0.2 = top 20%)
        days_to_expiry_range : tuple
            Target (min, max) days to expiration for options
        target_delta : float
            Target delta for put options (-0.3 = slightly OTM)
        min_open_interest : int
            Minimum open interest for option liquidity filter
        weekly_trade_day : int
            Day of week for trading (0=Monday, 4=Friday)
        store_intermediate : bool
            Whether to store intermediate results
        """
        # Define the factors required by this strategy
        factor_list = [
            'cum_return_252d_offset_21d',  # Momentum
            'cash_flow_to_asset',          # Quality
            'earnings_to_price',           # Value
            'inv_4q_growth'                # Investment growth
        ]
        
        # Call parent constructor with factor list
        super().__init__(factor_list)
        
        # Store strategy parameters
        self.top_pct = top_pct
        self.days_to_expiry_range = days_to_expiry_range
        self.target_delta = target_delta
        self.min_open_interest = min_open_interest
        self.weekly_trade_day = weekly_trade_day
        self.store_intermediate = store_intermediate
        
        # Initial weight for each position
        self.position_weight = 0.05
        
        # Storage for intermediate results
        self.intermediate_results = {}
    
    def is_trading_day(self, date):
        """Check if this is a rebalancing day"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return date.weekday() == self.weekly_trade_day
    
    def generate_option_targets(self, date, data_df, db_conn, min_price=5.0, 
                               market_cap_percentile=0.5, position_count=10, logger=print):
        """
        Generate option targets for put writing strategy
        
        Parameters:
        -----------
        date : datetime or str
            Current date
        data_df : DataFrame
            Stock data with factors and option availability flags
        db_conn : duckdb.DuckDBPyConnection
            Database connection for option data
        min_price : float
            Minimum price for underlying stocks
        market_cap_percentile : float
            Market cap percentile threshold
        position_count : int
            Number of put positions to generate
        logger : function
            Logging function
            
        Returns:
        --------
        dict : Option targets
        """
        # Reset intermediate storage
        if self.store_intermediate:
            self.intermediate_results = {}
            
        # Format date if needed
        if isinstance(date, str):
            date_obj = pd.to_datetime(date)
            date_str = date
        else:
            date_obj = date
            date_str = date.strftime('%Y-%m-%d')
        
        # Check if this is a rebalancing day
        is_rebalance = self.is_trading_day(date_obj)
        
        # Initialize targets
        targets = {
            'short_puts': [],
            'rebalance': is_rebalance
        }
        
        # If not a rebalancing day, return empty targets
        if not is_rebalance:
            logger(f"Not a rebalancing day: {date_str}")
            return targets
        
        # Filter universe for stock selection
        universe_df = data_df[
            (data_df['dlyprc'] > min_price) &
            (data_df['dlycap'] > data_df['dlycap'].quantile(market_cap_percentile)) &
            (data_df['option_available'] == 1) &  # Only stocks with options available
            (data_df['has_puts'] == 1)            # Specifically need puts
        ].copy()
        
        # Track intermediate result
        if self.store_intermediate:
            self.intermediate_results['initial_universe'] = universe_df.copy()
            
        logger(f"Starting universe: {len(universe_df)} stocks with options")
        
        # Check if we have enough stocks
        if len(universe_df) < 2:
            logger("Not enough stocks with options available")
            return targets
        
        # STAGE 1: Rank stocks by combined factors
        logger("Stage 1: Ranking stocks by factors")
        
        # Check if required factors exist
        available_factors = [f for f in self.factor_list if f in universe_df.columns]
        if not available_factors:
            logger("Error: No required factors available")
            return targets
        
        # Rank stocks by each factor
        for factor in available_factors:
            rank_col = f"{factor}_rank"
            # Most factors - higher is better, but inv_4q_growth - lower is better
            ascending = False if factor != 'inv_4q_growth' else True
            universe_df[rank_col] = universe_df[factor].rank(ascending=ascending)
        
        # Calculate combined rank 
        rank_columns = [f"{f}_rank" for f in available_factors]
        universe_df['combined_rank'] = universe_df[rank_columns].sum(axis=1)
        
        # Sort by combined rank
        universe_df.sort_values('combined_rank', ascending=True, inplace=True)
        
        # Track intermediate result
        if self.store_intermediate:
            self.intermediate_results['ranked_universe'] = universe_df.copy()
        
        # STAGE 2: Select top stocks
        logger("Stage 2: Selecting top stocks")
        
        # Calculate top stock count
        stock_count = len(universe_df)
        top_count = max(1, min(position_count*2, int(stock_count * self.top_pct)))
        
        # Select top stocks
        top_stocks = universe_df.iloc[:top_count]
        
        logger(f"Selected {len(top_stocks)} top stocks")
        
        # Track intermediate results
        if self.store_intermediate:
            self.intermediate_results['top_stocks'] = top_stocks.copy()
        
        # STAGE 3: Find put options for selected stocks
        logger("Stage 3: Finding put options")
        
        # Import required functions
        from option_tools import get_option_data_for_stock, select_option_by_delta
        
        # Parameters for option filtering
        delta_ranges = {
            'P': (self.target_delta - 0.1, self.target_delta + 0.1)
        }
        
        # Adjust position weight based on count (aim for equal weights)
        self.position_weight = 1.0 / min(position_count, len(top_stocks))
        
        # Process top stocks for put writing
        short_puts = []
        processed_puts = 0
        
        for _, row in top_stocks.iterrows():
            if processed_puts >= position_count:
                break
                
            permno = row.name
            secid = row.get('secid')
            
            # Skip stocks without valid secid
            if pd.isna(secid) or secid <= 0:
                continue
                
            # Get options for this stock
            options = get_option_data_for_stock(
                date_str=date_str,
                permno=permno,
                db_conn=db_conn,
                days_to_expiry_range=self.days_to_expiry_range,
                delta_ranges=delta_ranges,
                min_open_interest=self.min_open_interest,
                option_type='P'  # Only interested in puts
            )
            
            if len(options) == 0:
                continue
                
            # Find best put option
            best_put = select_option_by_delta(
                options,
                target_delta=self.target_delta,
                target_expiry_days=self.days_to_expiry_range[0] + (self.days_to_expiry_range[1] - self.days_to_expiry_range[0])//2,
                option_type='P'
            )
            
            if best_put is not None:
                # Add to targets
                short_puts.append({
                    'optionid': best_put['optionid'],
                    'permno': permno,
                    'strike': best_put['strike_price'],
                    'expiry': best_put['exdate'],
                    'weight': self.position_weight,
                    'hold_to_maturity': True,
                    'target_date': None
                })
                
                processed_puts += 1
        
        # Add to targets
        targets['short_puts'] = short_puts
        
        # Log summary
        logger(f"Final option targets: {len(short_puts)} short puts")
        
        self.save_intermediate_results(date_obj)
        return targets
    
    def get_intermediate_results(self):
        """Get stored intermediate results"""
        if not self.store_intermediate:
            return {"error": "Intermediate result storage is not enabled"}
        return self.intermediate_results
    
    def save_intermediate_results(self, date, directory="option_intermediate_results"):
        """Save intermediate results to disk"""
        import os
        
        if not self.store_intermediate:
            return
            
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        date_str = date.strftime('%Y%m%d')
        
        # Save each DataFrame
        for stage, df in self.intermediate_results.items():
            if isinstance(df, pd.DataFrame):
                filename = f"{directory}/{date_str}_{stage}.csv"
                df.to_csv(filename)