# option_tools.py
# Helper functions for option backtesting

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import duckdb
from tqdm import tqdm

from factor_tools import *


def get_option_table_name(date_str):
    """
    Get appropriate option table name based on date
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    str : Option table name (e.g., 'opprcd2020')
    """
    year = pd.to_datetime(date_str).year
    return f"opprcd{year}"


def load_option_links(date_str, db_path):
    """
    Load permno to secid links for a given date
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format
    db_path : str
        Path to the database file
        
    Returns:
    --------
    dict : {permno: [secid1, secid2, ...]} mapping
    """
    try:
        # Create a dedicated connection for this function
        with duckdb.connect(db_path, read_only=True) as db_conn:
            # Query PERMNO to SECID links for the given date
            query = f"""
            SELECT 
                permno,
                secid,
                sdate,
                edate
            FROM opcrsphist
            WHERE DATE '{date_str}' BETWEEN sdate AND edate
            """
            
            links_df = db_conn.execute(query).fetchdf()
            
            # Group by permno
            permno_to_secids = {}
            for _, row in links_df.iterrows():
                permno = row['permno']
                secid = row['secid']
                
                if permno not in permno_to_secids:
                    permno_to_secids[permno] = []
                    
                permno_to_secids[permno].append(secid)
            
            return permno_to_secids
        
    except Exception as e:
        print(f"Error loading option links: {str(e)}")
        return {}
    


def get_filtered_options(date_str, db_path, permnos=None, secids=None, min_open_interest=10,
                         min_volume=0, min_price=0.10, days_to_expiry_range=(15, 45),
                         delta_ranges=None, option_type=None):
    """
    Get a filtered list of options based on criteria including delta ranges
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format
    db_path : str
        Path to the database file
    permnos : list, optional
        List of permnos to filter options for
    secids : list, optional
        List of secids to filter options for (alternative to permnos)
    min_open_interest : int
        Minimum open interest for liquidity filter
    min_volume : int
        Minimum volume for liquidity filter
    min_price : float
        Minimum price for liquidity filter
    days_to_expiry_range : tuple
        (min_days, max_days) range for expiration dates
    delta_ranges : dict, optional
        Delta ranges to filter by, e.g. {'C': (0.4, 0.6), 'P': (-0.6, -0.4)}
    option_type : str, optional
        'C' for calls only, 'P' for puts only, None for both
        
    Returns:
    --------
    DataFrame : Filtered options data
    """
    # Get option table name based on year
    year = pd.to_datetime(date_str).year
    option_table = f"opprcd{year}"
    
    # Create a dedicated connection for this function
    with duckdb.connect(db_path, read_only=True) as db_conn:
        # Load option links if needed for permno filtering
        filter_clause = ""
        
        if permnos and not secids:
            # Query PERMNO to SECID links for the given date
            try:
                links_query = f"""
                SELECT 
                    permno,
                    secid
                FROM opcrsphist
                WHERE DATE '{date_str}' BETWEEN sdate AND edate
                AND permno IN ({','.join(map(str, permnos))})
                """
                
                links_df = db_conn.execute(links_query).fetchdf()
                secids = links_df['secid'].tolist()
            except Exception as e:
                print(f"Error loading option links: {str(e)}")
                return pd.DataFrame()
        
        if secids and len(secids) > 0:
            secid_str = ", ".join([str(s) for s in secids])
            filter_clause += f"AND o.secid IN ({secid_str})"
        
        # Calculate expiry date range
        current_date = pd.to_datetime(date_str)
        min_days, max_days = days_to_expiry_range
        
        min_date = (current_date + timedelta(days=min_days)).strftime('%Y-%m-%d')
        max_date = (current_date + timedelta(days=max_days)).strftime('%Y-%m-%d')
        
        # Add expiry date filter
        filter_clause += f" AND o.exdate BETWEEN DATE '{min_date}' AND DATE '{max_date}'"
        
        # Add option type filter if specified
        if option_type:
            filter_clause += f" AND o.cp_flag = '{option_type}'"
        
        # Add delta range filters if specified
        if delta_ranges:
            delta_conditions = []
            
            if 'C' in delta_ranges and (option_type is None or option_type == 'C'):
                min_delta, max_delta = delta_ranges['C']
                delta_conditions.append(f"(o.cp_flag = 'C' AND o.delta BETWEEN {min_delta} AND {max_delta})")
            
            if 'P' in delta_ranges and (option_type is None or option_type == 'P'):
                min_delta, max_delta = delta_ranges['P']
                delta_conditions.append(f"(o.cp_flag = 'P' AND o.delta BETWEEN {min_delta} AND {max_delta})")
            
            if delta_conditions:
                filter_clause += f" AND ({' OR '.join(delta_conditions)})"
        
        # Add liquidity filters
        filter_clause += f" AND o.open_interest >= {min_open_interest}"
        
        if min_volume > 0:
            filter_clause += f" AND o.volume >= {min_volume}"
        
        if min_price > 0:
            filter_clause += f" AND ((o.best_bid + o.best_offer) / 2) >= {min_price}"
        
        # Query to find options matching criteria - include contract_size
        query = f"""
        SELECT 
            o.optionid,
            o.secid,
            o.date,
            o.exdate,
            o.strike_price,
            o.cp_flag,
            o.best_bid AS bidprice,
            o.best_offer AS askprice,
            o.delta,
            o.gamma,
            o.vega,
            o.theta,
            o.impl_volatility,
            o.open_interest,
            o.volume,
            o.ss_flag,
            o.symbol,
            o.contract_size,  -- Include contract size
            l.permno
        FROM {option_table} o
        JOIN opcrsphist l ON o.secid = l.secid
        WHERE o.date = DATE '{date_str}'
        AND DATE '{date_str}' BETWEEN l.sdate AND l.edate
        {filter_clause}
        """
        
        try:
            filtered_options = db_conn.execute(query).fetchdf()
            
            # Calculate mid price
            if len(filtered_options) > 0:
                filtered_options['midprice'] = (filtered_options['bidprice'] + filtered_options['askprice']) / 2
                
                # Set default contract size if missing
                if 'contract_size' in filtered_options.columns:
                    filtered_options['contract_size'] = filtered_options['contract_size'].fillna(100)
                else:
                    filtered_options['contract_size'] = 100
            
            return filtered_options
        
        except Exception as e:
            print(f"Error getting filtered options: {str(e)}")
            return pd.DataFrame()
        

def select_option_by_delta(filtered_options, target_delta, target_expiry_days=30, option_type='C'):
    """
    Select a specific option from filtered options that matches closest to the desired 
    delta and expiry
    
    Parameters:
    -----------
    filtered_options : DataFrame
        Pre-filtered options data
    target_delta : float
        Desired delta value (e.g., 0.5 for calls, -0.5 for puts)
    target_expiry_days : int
        Target days to expiration
    option_type : str
        'C' for call, 'P' for put
        
    Returns:
    --------
    Series : The selected option, or None if no suitable option found
    """
    if len(filtered_options) == 0:
        return None
    
    # Filter by option type
    options = filtered_options[filtered_options['cp_flag'] == option_type].copy()
    
    if len(options) == 0:
        return None
    
    # Calculate days to expiry for each option
    current_date = pd.to_datetime(options['date'].iloc[0])
    options['days_to_expiry'] = (pd.to_datetime(options['exdate']) - current_date).dt.days
    
    # Find best expiry match
    options['expiry_diff'] = abs(options['days_to_expiry'] - target_expiry_days)
    min_expiry_diff = options['expiry_diff'].min()
    
    # Filter to options with closest expiry - create explicit copy
    exp_options = options[options['expiry_diff'] == min_expiry_diff].copy()
    selected_expiry = exp_options['exdate'].iloc[0]
    
    # From these options, find closest delta match
    exp_options['delta_diff'] = abs(exp_options['delta'] - target_delta)
    
    # Get the option with minimum delta difference
    best_option = exp_options.loc[exp_options['delta_diff'].idxmin()]
    
    return best_option

def find_straddle_pair(filtered_options, target_call_delta=0.5, target_put_delta=-0.5, 
                      target_expiry_days=30, pair_by_strike=True):
    """
    Find a call and put pair for a straddle strategy
    
    Parameters:
    -----------
    filtered_options : DataFrame
        Pre-filtered options data
    target_call_delta : float
        Target delta for call option
    target_put_delta : float
        Target delta for put option
    target_expiry_days : int
        Target days to expiration
    pair_by_strike : bool
        If True, ensure call and put have the same strike
        
    Returns:
    --------
    dict : Dictionary with selected call and put options
    """
    # Find best call and put individually
    best_call = select_option_by_delta(
        filtered_options, 
        target_delta=target_call_delta,
        target_expiry_days=target_expiry_days,
        option_type='C'
    )
    
    best_put = select_option_by_delta(
        filtered_options, 
        target_delta=target_put_delta,
        target_expiry_days=target_expiry_days,
        option_type='P'
    )
    
    if best_call is None or best_put is None:
        result = {}
        if best_call is not None:
            result['call'] = best_call
        if best_put is not None:
            result['put'] = best_put
        return result
    
    # If not pairing by strike, just return the individual best options
    if not pair_by_strike:
        return {
            'call': best_call,
            'put': best_put,
            'expiry': best_call['exdate']
        }
    
    # If strikes don't match, find options with matching strikes
    if best_call['strike_price'] != best_put['strike_price']:
        # Filter to the selected expiry
        
        exp_options = filtered_options[filtered_options['exdate'] == best_call['exdate']].copy()
        calls = exp_options[exp_options['cp_flag'] == 'C'].copy()
        puts = exp_options[exp_options['cp_flag'] == 'P'].copy()

        # Find common strikes
        call_strikes = set(calls['strike_price'])
        put_strikes = set(puts['strike_price'])
        common_strikes = call_strikes & put_strikes
        
        if common_strikes:
            # Find strikes closest to the average of our best call and put strikes
            avg_strike = (best_call['strike_price'] + best_put['strike_price']) / 2
            best_common_strike = min(common_strikes, key=lambda x: abs(x - avg_strike))
            
            # Get best call and put at this strike
            strike_calls = calls[calls['strike_price'] == best_common_strike]
            strike_puts = puts[puts['strike_price'] == best_common_strike]
            
            # Find options with closest deltas
            if not strike_calls.empty and not strike_puts.empty:
                strike_calls['delta_diff'] = abs(strike_calls['delta'] - target_call_delta)
                strike_puts['delta_diff'] = abs(strike_puts['delta'] - target_put_delta)
                
                best_call = strike_calls.loc[strike_calls['delta_diff'].idxmin()]
                best_put = strike_puts.loc[strike_puts['delta_diff'].idxmin()]
    
    # Return the paired options
    return {
        'call': best_call,
        'put': best_put,
        'strike': best_call['strike_price'],
        'expiry': best_call['exdate']
    }

def rank_stocks_by_factors(stock_data, factors, weights=None, top_pct=0.2, bottom_pct=0.2):
    """
    Rank stocks by multiple factors
    
    Parameters:
    -----------
    stock_data : DataFrame
        Daily stock data including factor columns
    factors : list
        List of factor names to use for ranking
    weights : dict, optional
        Dictionary of factor weights {factor: weight}
    top_pct : float
        Percentile cutoff for top stocks
    bottom_pct : float
        Percentile cutoff for bottom stocks
        
    Returns:
    --------
    tuple : (top_permnos, bottom_permnos, combined_ranks)
    """
    # Verify all factors exist
    missing_factors = [f for f in factors if f not in stock_data.columns]
    if missing_factors:
        print(f"Warning: Factors not found in data: {missing_factors}")
        factors = [f for f in factors if f in stock_data.columns]
        
    if not factors:
        return [], [], pd.DataFrame()
    
    # Default equal weights if not specified
    if weights is None:
        weights = {factor: 1/len(factors) for factor in factors}
    
    # Filter active stocks with valid factor values
    active_stocks = stock_data[stock_data['tradingstatusflg'] == 'A'].copy()
    
    # Only include stocks with all factors present
    for factor in factors:
        active_stocks = active_stocks[~pd.isna(active_stocks[factor])]
    
    if len(active_stocks) == 0:
        return [], [], pd.DataFrame()
    
    # Calculate percentile ranks for each factor
    # For some factors higher is better, for others lower is better
    # This can be controlled by the weight sign
    factor_ranks = pd.DataFrame(index=active_stocks.index)
    
    for factor in factors:
        # Determine if ascending (negative weight) or descending (positive weight)
        is_ascending = weights.get(factor, 1) < 0
        weight_sign = -1 if is_ascending else 1
        
        # Calculate percentile rank
        factor_ranks[f"{factor}_rank"] = active_stocks[factor].rank(
            ascending=is_ascending, 
            pct=True
        ) * weight_sign * abs(weights.get(factor, 1))
    
    # Calculate combined rank
    factor_ranks['combined_rank'] = factor_ranks.sum(axis=1)
    
    # Sort by combined rank (higher is better)
    sorted_ranks = factor_ranks.sort_values('combined_rank', ascending=False)
    
    # Get top and bottom stocks
    num_stocks = len(sorted_ranks)
    top_count = int(num_stocks * top_pct)
    bottom_count = int(num_stocks * bottom_pct)
    
    top_stocks = sorted_ranks.iloc[:top_count]
    bottom_stocks = sorted_ranks.iloc[-bottom_count:]
    
    # Return permnos and combined ranks
    return top_stocks.index.tolist(), bottom_stocks.index.tolist(), factor_ranks

'''
def load_option_by_id(date_str, optionid, db_path):
    """
    Load full data for a specific option
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format
    optionid : int
        Option ID
    db_path : str
        Path to the database file
        
    Returns:
    --------
    Series : Option data or None if not found
    """
    try:
        # Create a dedicated connection for this function
        with duckdb.connect(db_path, read_only=True) as db_conn:
            # Get the appropriate option table
            option_table = get_option_table_name(date_str)
            
            # Query specific option
            query = f"""
            SELECT 
                o.secid,
                o.date,
                o.optionid,
                o.exdate,
                o.strike_price,
                o.cp_flag,
                o.best_bid AS bidprice,
                o.best_offer AS askprice,
                o.impl_volatility,
                o.delta,
                o.gamma,
                o.vega,
                o.theta,
                o.open_interest,
                o.volume,
                o.ss_flag,
                o.symbol,
                o.am_settlement,
                o.contract_size,  -- Include contract size
                o.cfadj,          -- Include adjustment factor
                l.permno
            FROM {option_table} o
            JOIN opcrsphist l ON o.secid = l.secid
            WHERE o.date = DATE '{date_str}'
            AND o.optionid = {optionid}
            AND DATE '{date_str}' BETWEEN l.sdate AND l.edate
            """
            
            result = db_conn.execute(query).fetchdf()
            
            if len(result) == 0:
                return None
                
            # Calculate mid price
            result['midprice'] = (result['bidprice'] + result['askprice']) / 2
            
            # Set default contract size if missing
            if 'contract_size' in result.columns:
                result['contract_size'] = result['contract_size'].fillna(100)
            else:
                result['contract_size'] = 100
                
            return result.iloc[0]
        
    except Exception as e:
        print(f"Error loading option by ID: {str(e)}")
        return None
    


def load_option_batch(date_str, optionids, db_path):
    """
    Load full data for a batch of options
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format
    optionids : list
        List of option IDs
    db_path : str
        Path to the database file
        
    Returns:
    --------
    DataFrame : Option data for the requested IDs
    """
    if not optionids or len(optionids) == 0:
        return pd.DataFrame()
        
    try:
        # Create a dedicated connection for this function
        with duckdb.connect(db_path, read_only=True) as db_conn:
            # Get the appropriate option table
            option_table = get_option_table_name(date_str)
            
            # Format optionids for SQL IN clause
            optionids_str = ", ".join([str(oid) for oid in optionids])
            
            # Query multiple options
            query = f"""
            SELECT 
                o.secid,
                o.date,
                o.optionid,
                o.exdate,
                o.strike_price,
                o.cp_flag,
                o.best_bid AS bidprice,
                o.best_offer AS askprice,
                o.impl_volatility,
                o.delta,
                o.gamma,
                o.vega,
                o.theta,
                o.open_interest,
                o.volume,
                o.ss_flag,
                o.symbol,
                o.am_settlement,
                o.contract_size,  -- Include contract size
                o.cfadj,          -- Include adjustment factor
                l.permno
            FROM {option_table} o
            JOIN opcrsphist l ON o.secid = l.secid
            WHERE o.date = DATE '{date_str}'
            AND o.optionid IN ({optionids_str})
            AND DATE '{date_str}' BETWEEN l.sdate AND l.edate
            """
            
            result = db_conn.execute(query).fetchdf()
            
            # Calculate mid price
            if len(result) > 0:
                result['midprice'] = (result['bidprice'] + result['askprice']) / 2
                
                # Set default contract size if missing
                if 'contract_size' in result.columns:
                    result['contract_size'] = result['contract_size'].fillna(100)
                else:
                    result['contract_size'] = 100
                
            return result
        
    except Exception as e:
        print(f"Error loading option batch: {str(e)}")
        return pd.DataFrame()
    
'''

def calculate_option_price(option_data, is_opening=True, is_long=True):
    """
    Calculate appropriate price for an option trade
    
    Parameters:
    -----------
    option_data : Series or dict
        Option data with bid and ask prices
    is_opening : bool
        Whether this is an opening trade
    is_long : bool
        Whether this is a long position
        
    Returns:
    --------
    tuple : (price_per_share, price_per_contract, contract_size)
    """
    bid = option_data.get('bidprice', 0)
    ask = option_data.get('askprice', 0)
    
    # Use midprice if available
    if 'midprice' in option_data:
        mid = option_data['midprice']
    else:
        # Calculate mid price
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
    
    # For opening long positions or closing short positions, use ask price (paying more)
    # For opening short positions or closing long positions, use bid price (receiving less)
    if (is_opening and is_long) or (not is_opening and not is_long):
        # Buy at ask
        price_per_share = ask if ask > 0 else mid
    else:
        # Sell at bid
        price_per_share = bid if bid > 0 else mid
    
    # Get contract size (default to 100 if not available)
    contract_size = option_data.get('contract_size', 100)
    if pd.isna(contract_size) or contract_size <= 0:
        contract_size = 100
    
    # Calculate price per contract
    price_per_contract = price_per_share * contract_size
    
    return price_per_share, price_per_contract, contract_size

def calculate_contracts(weight, option_price, portfolio_value, contract_size=100):
    """
    Calculate number of contracts based on weight
    
    Parameters:
    -----------
    weight : float
        Target weight in portfolio (0-1)
    option_price : float
        Price per share of the option
    portfolio_value : float
        Current portfolio value
    contract_size : float
        Number of shares per contract
        
    Returns:
    --------
    int : Number of contracts to trade (rounded down)
    """
    if option_price <= 0 or contract_size <= 0:
        return 0
        
    # Calculate position value
    position_value = portfolio_value * weight
    
    # Convert to contracts using the actual contract size
    contract_value = option_price * contract_size
    contracts = int(position_value / contract_value)
    
    return max(0, contracts)



def create_option_base_data(start_date, end_date, db_path, min_option_liquidity=10):
    """
    Create base dataset that includes stock data plus information about 
    option availability (without loading full option data or factors).
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    db_path : str
        Path to the database file
    min_option_liquidity : int
        Minimum open interest for options to be considered available
        
    Returns:
    --------
    tuple : (data_df_dic, trading_dates)
    """

    
    db_conn = duckdb.connect(db_path, read_only=True)
    
    try:
        # Get previous trading date
        prev_date_query = f"""
            SELECT MAX(caldt) as prev_date
            FROM metaexchangecalendar
            WHERE tradingflg = 'Y'
            And caldt < DATE '{start_date}'
        """
        prev_start_date = db_conn.execute(prev_date_query).fetchone()[0]
        
        trading_dates = get_trading_dates(prev_start_date, end_date, db_conn)

        cache_dir = os.path.join('Data_all', 'Option_Base_Data')
        parquet_path = os.path.join(cache_dir, '{}.parquet')
        os.makedirs(cache_dir, exist_ok=True)

        data_df_dic = {}
        
        for date in tqdm(trading_dates, desc="Processing option base data"):
            try:
                date_str = pd.to_datetime(date).strftime('%Y%m%d')
                formatted_date = pd.to_datetime(date).strftime('%Y-%m-%d')
                
                # Check if data already exists
                if os.path.exists(parquet_path.format(date_str)):
                    data_df_dic[date] = pd.read_parquet(parquet_path.format(date_str))
                    continue
                
                # Get extended stock data
                stock_df = sql_extended_base_data(date, os.path.join('Data_all', 'Extended_Base_Data', '{}.parquet'), db_conn)
                
                # Query option availability (lightweight)
                query = f"""
                SELECT 
                    l.permno,
                    l.secid,
                    COUNT(DISTINCT o.optionid) as option_count,
                    MAX(CASE WHEN o.cp_flag = 'C' AND o.open_interest >= {min_option_liquidity} THEN 1 ELSE 0 END) as has_calls,
                    MAX(CASE WHEN o.cp_flag = 'P' AND o.open_interest >= {min_option_liquidity} THEN 1 ELSE 0 END) as has_puts,
                    MIN(CASE WHEN o.cp_flag = 'C' THEN o.strike_price ELSE NULL END) as min_call_strike,
                    MAX(CASE WHEN o.cp_flag = 'C' THEN o.strike_price ELSE NULL END) as max_call_strike,
                    AVG(CASE WHEN o.cp_flag = 'C' THEN o.impl_volatility ELSE NULL END) as avg_call_iv,
                    AVG(CASE WHEN o.cp_flag = 'P' THEN o.impl_volatility ELSE NULL END) as avg_put_iv,
                    MIN(o.exdate) as nearest_expiry,
                    MAX(o.exdate) as furthest_expiry
                FROM {get_option_table_name(formatted_date)} o
                JOIN opcrsphist l ON o.secid = l.secid
                WHERE o.date = DATE '{formatted_date}'
                AND DATE '{formatted_date}' BETWEEN l.sdate AND l.edate
                AND o.open_interest >= {min_option_liquidity}
                GROUP BY l.permno, l.secid
                """
                
                try:
                    option_avail_df = db_conn.execute(query).fetchdf()
                    
                    if len(option_avail_df) > 0:
                        # Process option availability data
                        option_avail_df = option_avail_df.set_index('permno')
                        
                        # Calculate days to expiry
                        current_date = pd.to_datetime(formatted_date)
                        option_avail_df['days_to_nearest_expiry'] = (pd.to_datetime(option_avail_df['nearest_expiry']) - current_date).dt.days
                        option_avail_df['days_to_furthest_expiry'] = (pd.to_datetime(option_avail_df['furthest_expiry']) - current_date).dt.days
                        
                        # Join option availability info to stock data
                        stock_df = stock_df.join(option_avail_df, how='left')
                        
                        # Fill NA values for option info
                        stock_df['secid'] = stock_df['secid'].fillna(-1).astype('Int64')
                        stock_df['option_count'] = stock_df['option_count'].fillna(0).astype('Int64')
                        stock_df['has_calls'] = stock_df['has_calls'].fillna(0).astype('Int64')
                        stock_df['has_puts'] = stock_df['has_puts'].fillna(0).astype('Int64')
                        
                        # Add option_available flag
                        stock_df['option_available'] = ((stock_df['has_calls'] > 0) & (stock_df['has_puts'] > 0)).astype(int)
                    else:
                        # Add empty columns if no options data found
                        stock_df['secid'] = -1
                        stock_df['option_count'] = 0
                        stock_df['has_calls'] = 0
                        stock_df['has_puts'] = 0
                        stock_df['option_available'] = 0
                        
                except Exception as e:
                    print(f"Error querying option availability: {str(e)}")
                    # Create empty columns if query fails
                    stock_df['secid'] = -1
                    stock_df['option_count'] = 0
                    stock_df['has_calls'] = 0
                    stock_df['has_puts'] = 0
                    stock_df['option_available'] = 0
                
                # Store the dataframe
                data_df_dic[date] = stock_df
                
                # Save data for future use
                data_df_dic[date].to_parquet(parquet_path.format(date_str))
                    
            except Exception as e:
                print(f"\nError processing date {date}: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        return data_df_dic, trading_dates

    finally:
        db_conn.close()


def get_option_data_for_stock_fromdb(date_str, permno, db_path, days_to_expiry_range=(15, 45), 
                             delta_ranges=None, min_open_interest=10, option_type=None):
    """
    Get option data for a specific stock
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format
    permno : int
        Stock PERMNO
    db_path : str
        Path to the database file
    days_to_expiry_range : tuple
        Target (min, max) days to expiration
    delta_ranges : dict, optional
        Delta ranges to filter by, e.g. {'C': (0.4, 0.6), 'P': (-0.6, -0.4)}
    min_open_interest : int
        Minimum open interest
    option_type : str, optional
        'C' for calls only, 'P' for puts only, None for both
    
    Returns:
    --------
    DataFrame : Filtered options data for the stock
    """

    
    # Create a dedicated connection for this function
    with duckdb.connect(db_path, read_only=True) as db_conn:
        # Check if we have secid information
        secid_query = f"""
        SELECT secid
        FROM opcrsphist
        WHERE permno = {permno}
        AND DATE '{date_str}' BETWEEN sdate AND edate
        """
        
        secid_df = db_conn.execute(secid_query).fetchdf()
        
        if len(secid_df) == 0:
            return pd.DataFrame()  # No options available
        
        secids = secid_df['secid'].tolist()
    
    # Now use get_filtered_options with these secids but pass db_path instead of db_conn
    return get_filtered_options(
        date_str=date_str,
        db_path=db_path,
        secids=secids,
        min_open_interest=min_open_interest,
        days_to_expiry_range=days_to_expiry_range,
        delta_ranges=delta_ranges,
        option_type=option_type
    )



def create_option_analysis_data(start_date, end_date, factor_names, db_path, factor_data_path,
                             option_base_path='Data_all/Option_Base_Data'):
    """
    Create analysis dataset that combines option base data with factors.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    factor_names : list
        List of factors to include
    db_path : str
        Path to the database file
    factor_data_path : str
        Base path to factor data
    option_base_path : str
        Path to the option base data directory
        
    Returns:
    --------
    tuple : (data_df_dic, trading_dates)
    """
    
    if isinstance(factor_names, str):
        factor_names = [factor_names]
        
    db_conn = duckdb.connect(db_path, read_only=True)
    
    try:
        # Get previous trading date
        prev_date_query = f"""
            SELECT MAX(caldt) as prev_date
            FROM metaexchangecalendar
            WHERE tradingflg = 'Y'
            And caldt < DATE '{start_date}'
        """
        prev_start_date = db_conn.execute(prev_date_query).fetchone()[0]
        
        # Get trading dates for the period
        trading_dates = get_trading_dates(prev_start_date, end_date, db_conn)
        
        # Create output directory with factors encoded in the path
        factor_str = "_".join(sorted(factor_names))
        cache_dir = os.path.join('Data_all', 'Option_Analysis_Data', factor_str)
        parquet_path = os.path.join(cache_dir, '{}.parquet')
        os.makedirs(cache_dir, exist_ok=True)
        
        data_df_dic = {}
        
        for date in tqdm(trading_dates, desc=f"Creating analysis data with factors: {factor_str}"):
            try:
                date_str = pd.to_datetime(date).strftime('%Y%m%d')
                
                # Check if data already exists
                if os.path.exists(parquet_path.format(date_str)):
                    data_df_dic[date] = pd.read_parquet(parquet_path.format(date_str))
                    continue
                
                # Path to base data
                base_data_path = os.path.join(option_base_path, f"{date_str}.parquet")
                
                # Check if base data exists
                if not os.path.exists(base_data_path):
                    print(f"Warning: No base data for date {date}, skipping...")
                    continue
                
                # Load base data
                base_df = pd.read_parquet(base_data_path)
                
                # Load factor data
                factor_df = load_factor_data(date, factor_names, factor_data_path)
                
                if factor_df is not None and len(factor_df) > 0:
                    # Ensure permno is available for merging
                    if 'permno' not in factor_df.columns and factor_df.index.name != 'permno':
                        if factor_df.index.name is None:
                            # Assume index is permno
                            factor_df = factor_df.reset_index().rename(columns={'index': 'permno'})
                        else:
                            # Use index name
                            factor_df = factor_df.reset_index()
                    
                    # Reset index on base_df if needed
                    base_df_reset = base_df.reset_index() if base_df.index.name == 'permno' else base_df
                    
                    # Reset index on factor_df if needed
                    factor_df_reset = factor_df.reset_index() if (factor_df.index.name == 'permno' and 'permno' not in factor_df.columns) else factor_df
                    
                    # Ensure both DataFrames have permno as a column
                    if 'permno' not in base_df_reset.columns:
                        print(f"Warning: permno column missing in base data for {date}")
                        continue
                        
                    if 'permno' not in factor_df_reset.columns:
                        print(f"Warning: permno column missing in factor data for {date}")
                        continue
                    
                    # Remove duplicate columns (except permno)
                    duplicate_cols = set(base_df_reset.columns) & set(factor_df_reset.columns) - {'permno'}
                    if duplicate_cols:
                        factor_df_reset = factor_df_reset.drop(columns=list(duplicate_cols))
                    
                    # Merge on permno column
                    merged_df = pd.merge(
                        base_df_reset,
                        factor_df_reset,
                        on='permno',
                        how='left'  # keep all stocks in base_df
                    )
                    
                    # Set permno back as index if it was originally
                    if base_df.index.name == 'permno':
                        merged_df.set_index('permno', inplace=True)
                    
                    # Store the merged dataframe
                    data_df_dic[date] = merged_df
                else:
                    # If no factor data, just use the base data
                    data_df_dic[date] = base_df
                
                # Save data for future use
                data_df_dic[date].to_parquet(parquet_path.format(date_str))
                    
            except Exception as e:
                print(f"\nError processing date {date}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        return data_df_dic, trading_dates
        
    finally:
        db_conn.close()


def preload_option_data_for_backtest(strategy, data_df_dic, trading_dates, db_path, 
                                    start_date=None, end_date=None, cache_dir='Data_all/Option_Cache_Data'):
    """
    Preload option data for a backtest, based on a strategy and date range.
    
    Parameters:
    -----------
    strategy : OptionStrategyBase
        The strategy that will be used for the backtest
    data_df_dic : dict
        Dictionary of stock data by date
    trading_dates : list
        List of trading dates for backtest
    db_path : str
        Path to the database
    start_date : datetime or str, optional
        Start date for the backtest (defaults to first trading date)
    end_date : datetime or str, optional
        End date for the backtest (defaults to last trading date)
    cache_dir : str
        Directory for caching option data
        
    Returns:
    --------
    dict : Dictionary of {date: option_dataframe} with preloaded data
    """


    
    # Use first and last trading dates if not specified
    if start_date is None:
        start_date = trading_dates[0]
    if end_date is None:
        end_date = trading_dates[-1]
    
    # Convert dates to datetime
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Filter trading dates to the specified range
    filtered_dates = [d for d in trading_dates if start_date <= d <= end_date]
    
    # First pass: Identify all option IDs we'll need
    print("First pass: Identifying required options...")
    option_ids_by_date = {}
    rebalance_dates = []
    
    for date in tqdm(filtered_dates, desc="Identifying options"):
        try:
            # Get stock data for current date
            if date not in data_df_dic:
                continue
                
            current_data_df = data_df_dic[date]
            
            # Check if this is a rebalance date (based on strategy)
            is_rebalance = strategy.is_trading_day(date)
            
            if not is_rebalance:
                continue
            
            # Store rebalance date for second pass
            rebalance_dates.append(date)
            
            # Generate option targets to identify needed options
            targets = strategy.generate_option_targets(
                date=date,
                data_df=current_data_df,
                db_path=db_path,
                logger=lambda x: None  # Silent logger
            )
            
            # Extract option IDs from targets
            date_option_ids = []
            
            # Process each category of options
            for category in ['long_calls', 'long_puts', 'short_calls', 'short_puts']:
                for option in targets.get(category, []):
                    option_id = option.get('optionid')
                    if option_id is not None:
                        date_option_ids.append(option_id)
            
            # Store unique option IDs for this date
            option_ids_by_date[date] = list(set(date_option_ids))
        
        except Exception as e:
            print(f"Error identifying options for {date}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            option_ids_by_date[date] = []
    
    # Second pass: Load option data for all identified option IDs
    print("Second pass: Loading option data...")
    all_option_ids = set()
    for date_option_ids in option_ids_by_date.values():
        all_option_ids.update(date_option_ids)
    
    print(f"Loading data for {len(all_option_ids)} unique option IDs")
    
    # Extract years from start and end dates
    start_year = start_date.year
    end_year = end_date.year
    
    # Group option IDs by year to query appropriate tables
    options_by_year = {}
    
    # Group options by relevant years
    for option_id in all_option_ids:
        for year in range(start_year, end_year + 1):
            options_by_year.setdefault(year, []).append(option_id)
    
    # Load option data for each year
    all_option_data = []
    
    for year, year_option_ids in options_by_year.items():
        option_table = f"opprcd{year}"
        
        # Split into chunks to avoid too large queries
        chunk_size = 1000  # Reduced chunk size to avoid memory issues
        option_id_chunks = [year_option_ids[i:i+chunk_size] for i in range(0, len(year_option_ids), chunk_size)]
        
        for chunk in tqdm(option_id_chunks, desc=f"Loading options from {option_table}"):
            try:
                with duckdb.connect(db_path, read_only=True) as db_conn:
                    option_id_str = ", ".join(str(oid) for oid in chunk)
                    
                    query = f"""
                    SELECT 
                        o.secid,
                        o.date,
                        o.optionid,
                        o.exdate,
                        o.strike_price,
                        o.cp_flag,
                        o.best_bid AS bidprice,
                        o.best_offer AS askprice,
                        o.impl_volatility,
                        o.delta,
                        o.gamma,
                        o.vega,
                        o.theta,
                        o.open_interest,
                        o.volume,
                        o.ss_flag,
                        o.symbol,
                        o.am_settlement,
                        o.contract_size,
                        o.cfadj,
                        l.permno
                    FROM {option_table} o
                    LEFT JOIN opcrsphist l ON o.secid = l.secid AND o.date BETWEEN l.sdate AND l.edate
                    WHERE o.optionid IN ({option_id_str})
                    AND o.date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                    """
                    
                    chunk_data = db_conn.execute(query).fetchdf()
                    
                    # Calculate mid price and set default contract size
                    if len(chunk_data) > 0:
                        chunk_data['midprice'] = (chunk_data['bidprice'] + chunk_data['askprice']) / 2
                        chunk_data['contract_size'] = chunk_data['contract_size'].fillna(100)
                    
                    all_option_data.append(chunk_data)
                    
                    # Clear variables to free memory
                    del chunk_data
            
            except Exception as e:
                print(f"Error loading options for {option_table} chunk: {str(e)}")
                import traceback
                print(traceback.format_exc())
    
    # Combine all option data
    if all_option_data:
        combined_option_data = pd.concat(all_option_data, ignore_index=True)
    else:
        combined_option_data = pd.DataFrame()
    
    # Convert date to datetime for consistency
    if len(combined_option_data) > 0 and 'date' in combined_option_data.columns:
        combined_option_data['date'] = pd.to_datetime(combined_option_data['date'])
    
    # Organize by date
    option_data_dic = {}
    for date in filtered_dates:
        date_data = combined_option_data[combined_option_data['date'] == date].copy() if len(combined_option_data) > 0 else pd.DataFrame()
        option_data_dic[date] = date_data
    
    # Clear large dataframes to free memory
    del combined_option_data
    
    return option_data_dic

'''
def load_option_by_id_from_preloaded(option_data_dic, date, optionid, db_path=None):
    """
    Load option data using preloaded data instead of querying the database.
    
    Parameters:
    -----------
    option_data_dic : dict
        Dictionary of preloaded option data by date
    date : str or datetime
        Date to lookup
    optionid : int
        Option ID to lookup
    db_path : str, optional
        Path to database for fallback queries
        
    Returns:
    --------
    Series : Option data or None if not found
    """
    import pandas as pd
    
    # Convert date to datetime for dictionary lookup
    if isinstance(date, str):
        date_obj = pd.to_datetime(date)
    else:
        date_obj = date
    
    # Check if we have data for this date
    if date_obj not in option_data_dic:
        # Fall back to database query if db_path provided
        if db_path:
            import option_tools as opt
            return opt.load_option_by_id(date, optionid, db_path)
        else:
            return None
    
    # Get date's option data
    date_options = option_data_dic[date_obj]
    
    # Find the specific option
    if len(date_options) == 0:
        # Fall back to database query if db_path provided
        if db_path:
            import option_tools as opt
            return opt.load_option_by_id(date, optionid, db_path)
        else:
            return None
    
    option_data = date_options[date_options['optionid'] == optionid]
    
    if len(option_data) == 0:
        # Option not found in preloaded data, fallback to loading from database
        if db_path:
            import option_tools as opt
            return opt.load_option_by_id(date, optionid, db_path)
        else:
            return None
    
    # Return the first matching row
    return option_data.iloc[0]


def load_option_batch_from_preloaded(option_data_dic, date, optionids, db_path=None):
    """
    Load batch of option data using preloaded data instead of querying the database.
    
    Parameters:
    -----------
    option_data_dic : dict
        Dictionary of preloaded option data by date
    date : str or datetime
        Date to lookup
    optionids : list
        List of option IDs to lookup
    db_path : str, optional
        Path to database for fallback queries
        
    Returns:
    --------
    DataFrame : Option data for requested IDs
    """
    import pandas as pd
    
    # Convert date to datetime for dictionary lookup
    if isinstance(date, str):
        date_obj = pd.to_datetime(date)
    else:
        date_obj = date
    
    # Check if we have data for this date
    if date_obj not in option_data_dic:
        # Fall back to database query if db_path provided
        if db_path:
            import option_tools as opt
            return opt.load_option_batch(date, optionids, db_path)
        else:
            return pd.DataFrame()
    
    # Get date's option data
    date_options = option_data_dic[date_obj]
    
    # Filter for requested option IDs
    if len(date_options) == 0:
        # Fall back to database query if db_path provided
        if db_path:
            import option_tools as opt
            return opt.load_option_batch(date, optionids, db_path)
        else:
            return pd.DataFrame()
    
    result = date_options[date_options['optionid'].isin(optionids)].copy()
    
    # If any options are missing and db_path is provided, fall back to database for those
    if len(result) < len(optionids) and db_path:
        # Some options not found in preloaded data
        missing_ids = set(optionids) - set(result['optionid'].tolist())
        
        # Fallback to loading from database for missing options
        import option_tools as opt
        missing_data = opt.load_option_batch(date, list(missing_ids), db_path)
        
        # Combine with preloaded data if any missing data was found
        if len(missing_data) > 0:
            result = pd.concat([result, missing_data], ignore_index=True)
    
    return result
'''

def get_option_data(date, optionid=None, optionids=None, permno=None, option_data_dic=None, db_path=None, **kwargs):
    """
    Universal function to get option data - always tries preloaded data first, falls back to database
    
    Parameters:
    -----------
    date : str or datetime
        Date to lookup
    optionid : int, optional
        Single option ID to retrieve (use either optionid or optionids, not both)
    optionids : list, optional
        List of option IDs to retrieve (use either optionid or optionids, not both)
    permno : int, optional
        PERMNO to filter options for (only used if optionid/optionids not provided)
    option_data_dic : dict
        Dictionary of preloaded option data {date: DataFrame}
    db_path : str
        Path to database for fallback queries
    **kwargs : dict
        Additional filtering parameters
        
    Returns:
    --------
    Series, DataFrame, or None : Option data based on the query type
    """
    import pandas as pd
    
    # Convert date for consistency
    if isinstance(date, str):
        date_obj = pd.to_datetime(date)
    else:
        date_obj = date
        date = date_obj.strftime('%Y-%m-%d')  # Create string version for database queries
    
    # Determine which retrieval mode we're in
    if optionid is not None:
        # Single option retrieval mode
        retrieval_mode = 'single'
    elif optionids is not None:
        # Batch option retrieval mode
        retrieval_mode = 'batch'
    elif permno is not None:
        # Stock options retrieval mode
        retrieval_mode = 'stock'
    else:
        raise ValueError("Must provide either optionid, optionids, or permno")
    
    # Check if we have preloaded data and the date exists in it
    if option_data_dic is not None and date_obj in option_data_dic:
        date_options = option_data_dic[date_obj]
        
        if len(date_options) > 0:
            # We have data for this date, attempt to use it
            
            if retrieval_mode == 'single':
                # Find the specific option
                option_data = date_options[date_options['optionid'] == optionid]
                
                if len(option_data) > 0:
                    return option_data.iloc[0]  # Return as Series
            
            elif retrieval_mode == 'batch':
                # Filter for requested option IDs
                result = date_options[date_options['optionid'].isin(optionids)].copy()
                
                if len(result) == len(optionids):
                    return result  # All options found
                
                # If we have some but not all options, fall through to database for missing ones
                if len(result) > 0:
                    # Find missing options
                    missing_ids = set(optionids) - set(result['optionid'].tolist())
                    
                    # Get missing options from database
                    if db_path is not None:
                        # Use direct database query for missing options - fix for recursion
                        # Use _db_load_option_batch instead of opt.load_option_batch to prevent recursion
                        missing_data = _db_load_option_batch(date, list(missing_ids), db_path)
                        
                        if len(missing_data) > 0:
                            # Combine with preloaded data
                            return pd.concat([result, missing_data], ignore_index=True)
                    
                    # If we can't get missing options, return what we have
                    return result
            
            elif retrieval_mode == 'stock':
                # Filter by permno
                stock_options = date_options[date_options['permno'] == permno].copy()
                
                if len(stock_options) > 0:
                    # Apply additional filters from kwargs
                    filtered_options = filter_options(stock_options, **kwargs)
                    
                    if len(filtered_options) > 0:
                        return filtered_options
    
    # If we reach here, either:
    # 1. We don't have preloaded data
    # 2. The date isn't in our preloaded data
    # 3. The specific option(s) weren't in the preloaded data
    # 4. Filtering removed all options
    
    # Fall back to database queries if db_path is provided
    if db_path is not None:
        if retrieval_mode == 'single':
            # Use direct database query
            return _db_load_option_by_id(date, optionid, db_path)
        
        elif retrieval_mode == 'batch':
            # Use direct database query
            return _db_load_option_batch(date, optionids, db_path)
        
        elif retrieval_mode == 'stock':
            # Use direct database query
            return _db_get_option_data_for_stock(
                date_str=date,
                permno=permno,
                db_path=db_path,
                **kwargs
            )
    
    # If no database path or nothing found
    if retrieval_mode == 'single':
        return None
    else:
        return pd.DataFrame()  # Empty DataFrame
    


def filter_options(options_df, days_to_expiry_range=None, delta_ranges=None, 
                  min_open_interest=None, option_type=None, **kwargs):
    """
    Apply filters to option data - used by get_option_data for filtering preloaded data
    
    Parameters:
    -----------
    options_df : DataFrame
        Options data to filter
    days_to_expiry_range : tuple, optional
        (min_days, max_days) range for expiration
    delta_ranges : dict, optional
        Delta ranges by option type {'C': (min_delta, max_delta), 'P': (min_delta, max_delta)}
    min_open_interest : int, optional
        Minimum open interest threshold
    option_type : str, optional
        'C' for calls only, 'P' for puts only
    **kwargs : dict
        Additional filters to apply
        
    Returns:
    --------
    DataFrame : Filtered options data
    """
    import pandas as pd
    
    # Make a copy to avoid modifying the original
    filtered = options_df.copy()
    
    # Filter by days to expiry
    if days_to_expiry_range is not None:
        min_days, max_days = days_to_expiry_range
        
        # If days_to_expiry not already calculated, calculate it
        if 'days_to_expiry' not in filtered.columns:
            # Get the date from the first row
            date = pd.to_datetime(filtered['date'].iloc[0])
            
            # Calculate days to expiry
            filtered['days_to_expiry'] = (pd.to_datetime(filtered['exdate']) - date).dt.days
        
        # Apply filter
        filtered = filtered[
            (filtered['days_to_expiry'] >= min_days) & 
            (filtered['days_to_expiry'] <= max_days)
        ]
    
    # Filter by option type
    if option_type is not None:
        if option_type in ['C', 'P']:
            filtered = filtered[filtered['cp_flag'] == option_type]
    
    # Filter by open interest
    if min_open_interest is not None:
        filtered = filtered[filtered['open_interest'] >= min_open_interest]
    
    # Filter by delta ranges
    if delta_ranges is not None:
        filtered_by_delta = []
        
        for opt_type, (min_delta, max_delta) in delta_ranges.items():
            # Filter by option type and delta range
            type_options = filtered[filtered['cp_flag'] == opt_type]
            
            if len(type_options) > 0:
                delta_filtered = type_options[
                    (type_options['delta'] >= min_delta) & 
                    (type_options['delta'] <= max_delta)
                ]
                
                if len(delta_filtered) > 0:
                    filtered_by_delta.append(delta_filtered)
        
        # Combine delta-filtered options
        if filtered_by_delta:
            filtered = pd.concat(filtered_by_delta, ignore_index=True)
        else:
            filtered = pd.DataFrame(columns=filtered.columns)  # Empty DataFrame with same columns
    
    return filtered


# 2. Replacement functions for the core option data access methods
def load_option_by_id(date, optionid, db_path=None, option_data_dic=None):
    """
    Load option data for a specific option ID, prioritizing preloaded data
    
    Parameters:
    -----------
    date : str or datetime
        Date to lookup
    optionid : int
        Option ID to retrieve
    db_path : str
        Path to database for fallback queries
    option_data_dic : dict
        Dictionary of preloaded option data
        
    Returns:
    --------
    Series or None : Option data if found
    """
    return get_option_data(
        date=date, 
        optionid=optionid, 
        option_data_dic=option_data_dic, 
        db_path=db_path
    )


def load_option_batch(date, optionids, db_path=None, option_data_dic=None):
    """
    Load option data for multiple option IDs, prioritizing preloaded data
    
    Parameters:
    -----------
    date : str or datetime
        Date to lookup
    optionids : list
        List of option IDs to retrieve
    db_path : str
        Path to database for fallback queries
    option_data_dic : dict
        Dictionary of preloaded option data
        
    Returns:
    --------
    DataFrame : Option data for requested IDs
    """
    return get_option_data(
        date=date, 
        optionids=optionids, 
        option_data_dic=option_data_dic, 
        db_path=db_path
    )


def get_option_data_for_stock(date_str, permno, db_path=None, option_data_dic=None, **kwargs):
    """
    Get option data for a specific stock, prioritizing preloaded data
    
    Parameters:
    -----------
    date_str : str
        Date in string format
    permno : int
        PERMNO of the stock
    db_path : str
        Path to database for fallback queries
    option_data_dic : dict
        Dictionary of preloaded option data
    **kwargs : dict
        Additional filtering parameters
        
    Returns:
    --------
    DataFrame : Option data for the specified stock
    """
    return get_option_data(
        date=date_str, 
        permno=permno, 
        option_data_dic=option_data_dic, 
        db_path=db_path, 
        **kwargs
    )

def _db_load_option_by_id(date_str, optionid, db_path):
    """
    Database-direct version that loads full data for a specific option (no recursion)
    """
    try:

        # Convert date if needed
        if isinstance(date_str, datetime):
            date_str = date_str.strftime('%Y-%m-%d')
        
        # Create a dedicated connection for this function
        with duckdb.connect(db_path, read_only=True) as db_conn:
            # Get the appropriate option table
            year = pd.to_datetime(date_str).year
            option_table = f"opprcd{year}"
            
            # Query specific option
            query = f"""
            SELECT 
                o.secid,
                o.date,
                o.optionid,
                o.exdate,
                o.strike_price,
                o.cp_flag,
                o.best_bid AS bidprice,
                o.best_offer AS askprice,
                o.impl_volatility,
                o.delta,
                o.gamma,
                o.vega,
                o.theta,
                o.open_interest,
                o.volume,
                o.ss_flag,
                o.symbol,
                o.am_settlement,
                o.contract_size,  -- Include contract size
                o.cfadj,          -- Include adjustment factor
                l.permno
            FROM {option_table} o
            JOIN opcrsphist l ON o.secid = l.secid
            WHERE o.date = DATE '{date_str}'
            AND o.optionid = {optionid}
            AND DATE '{date_str}' BETWEEN l.sdate AND l.edate
            """
            
            result = db_conn.execute(query).fetchdf()
            
            if len(result) == 0:
                return None
                
            # Calculate mid price
            result['midprice'] = (result['bidprice'] + result['askprice']) / 2
            
            # Set default contract size if missing
            if 'contract_size' in result.columns:
                result['contract_size'] = result['contract_size'].fillna(100)
            else:
                result['contract_size'] = 100
                
            return result.iloc[0]
        
    except Exception as e:
        print(f"Error loading option by ID: {str(e)}")
        return None


def _db_load_option_batch(date_str, optionids, db_path):
    """
    Database-direct version that loads full data for a batch of options (no recursion)
    """
    if not optionids or len(optionids) == 0:
        return pd.DataFrame()
        
    try:

        
        # Convert date if needed
        if isinstance(date_str, datetime):
            date_str = date_str.strftime('%Y-%m-%d')
        
        # Create a dedicated connection for this function
        with duckdb.connect(db_path, read_only=True) as db_conn:
            # Get the appropriate option table
            year = pd.to_datetime(date_str).year
            option_table = f"opprcd{year}"
            
            # Format optionids for SQL IN clause
            optionids_str = ", ".join([str(oid) for oid in optionids])
            
            # Query multiple options
            query = f"""
            SELECT 
                o.secid,
                o.date,
                o.optionid,
                o.exdate,
                o.strike_price,
                o.cp_flag,
                o.best_bid AS bidprice,
                o.best_offer AS askprice,
                o.impl_volatility,
                o.delta,
                o.gamma,
                o.vega,
                o.theta,
                o.open_interest,
                o.volume,
                o.ss_flag,
                o.symbol,
                o.am_settlement,
                o.contract_size,  -- Include contract size
                o.cfadj,          -- Include adjustment factor
                l.permno
            FROM {option_table} o
            JOIN opcrsphist l ON o.secid = l.secid
            WHERE o.date = DATE '{date_str}'
            AND o.optionid IN ({optionids_str})
            AND DATE '{date_str}' BETWEEN l.sdate AND l.edate
            """
            
            result = db_conn.execute(query).fetchdf()
            
            # Calculate mid price
            if len(result) > 0:
                result['midprice'] = (result['bidprice'] + result['askprice']) / 2
                
                # Set default contract size if missing
                if 'contract_size' in result.columns:
                    result['contract_size'] = result['contract_size'].fillna(100)
                else:
                    result['contract_size'] = 100
                
            return result
        
    except Exception as e:
        print(f"Error loading option batch: {str(e)}")
        return pd.DataFrame()


def _db_get_option_data_for_stock(date_str, permno, db_path, days_to_expiry_range=(15, 45), 
                                delta_ranges=None, min_open_interest=10, option_type=None):
    """
    Database-direct version that gets option data for a specific stock (no recursion)
    """
    try:

        
        # Convert date if needed
        if isinstance(date_str, datetime):
            date_str = date_str.strftime('%Y-%m-%d')
        
        # Create a dedicated connection for this function
        with duckdb.connect(db_path, read_only=True) as db_conn:
            # Check if we have secid information
            secid_query = f"""
            SELECT secid
            FROM opcrsphist
            WHERE permno = {permno}
            AND DATE '{date_str}' BETWEEN sdate AND edate
            """
            
            secid_df = db_conn.execute(secid_query).fetchdf()
            
            if len(secid_df) == 0:
                return pd.DataFrame()  # No options available
            
            secids = secid_df['secid'].tolist()
            
            # Get option table name based on year
            year = pd.to_datetime(date_str).year
            option_table = f"opprcd{year}"
            
            # Generate the filter clause for the query
            filter_clause = ""
            
            # Add secid filter
            if secids and len(secids) > 0:
                secid_str = ", ".join([str(s) for s in secids])
                filter_clause += f"AND o.secid IN ({secid_str})"
            
            # Calculate expiry date range
            current_date = pd.to_datetime(date_str)
            min_days, max_days = days_to_expiry_range
            
            min_date = (current_date + timedelta(days=min_days)).strftime('%Y-%m-%d')
            max_date = (current_date + timedelta(days=max_days)).strftime('%Y-%m-%d')
            
            # Add expiry date filter
            filter_clause += f" AND o.exdate BETWEEN DATE '{min_date}' AND DATE '{max_date}'"
            
            # Add option type filter if specified
            if option_type:
                filter_clause += f" AND o.cp_flag = '{option_type}'"
            
            # Add delta range filters if specified
            if delta_ranges:
                delta_conditions = []
                
                if 'C' in delta_ranges and (option_type is None or option_type == 'C'):
                    min_delta, max_delta = delta_ranges['C']
                    delta_conditions.append(f"(o.cp_flag = 'C' AND o.delta BETWEEN {min_delta} AND {max_delta})")
                
                if 'P' in delta_ranges and (option_type is None or option_type == 'P'):
                    min_delta, max_delta = delta_ranges['P']
                    delta_conditions.append(f"(o.cp_flag = 'P' AND o.delta BETWEEN {min_delta} AND {max_delta})")
                
                if delta_conditions:
                    filter_clause += f" AND ({' OR '.join(delta_conditions)})"
            
            # Add liquidity filters
            filter_clause += f" AND o.open_interest >= {min_open_interest}"
            
            # Query to find options matching criteria
            query = f"""
            SELECT 
                o.optionid,
                o.secid,
                o.date,
                o.exdate,
                o.strike_price,
                o.cp_flag,
                o.best_bid AS bidprice,
                o.best_offer AS askprice,
                o.delta,
                o.gamma,
                o.vega,
                o.theta,
                o.impl_volatility,
                o.open_interest,
                o.volume,
                o.ss_flag,
                o.symbol,
                o.contract_size,
                l.permno
            FROM {option_table} o
            JOIN opcrsphist l ON o.secid = l.secid
            WHERE o.date = DATE '{date_str}'
            AND DATE '{date_str}' BETWEEN l.sdate AND l.edate
            {filter_clause}
            """
            
            filtered_options = db_conn.execute(query).fetchdf()
            
            # Calculate mid price
            if len(filtered_options) > 0:
                filtered_options['midprice'] = (filtered_options['bidprice'] + filtered_options['askprice']) / 2
                
                # Set default contract size if missing
                if 'contract_size' in filtered_options.columns:
                    filtered_options['contract_size'] = filtered_options['contract_size'].fillna(100)
                else:
                    filtered_options['contract_size'] = 100
            
            return filtered_options
            
    except Exception as e:
        print(f"Error getting filtered options: {str(e)}")
        return pd.DataFrame()