# factor_tools.py

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent
from FactorCalculator import FactorNeutralizer
import concurrent.futures
from pathlib import Path


def get_all_investment_universe(date_str, duck_conn):
    """
    Define the complete investment universe for a given date based on CRSP-Compustat linkage.
    
    Args:
        date_str (str): Date in 'YYYY-MM-DD' format
        duck_conn: Active DuckDB connection
        
    Returns:
        DataFrame: Universe of stocks meeting our criteria
    """
    universe_query = """
    WITH valid_gvkeys AS (
        SELECT DISTINCT gvkey 
        FROM wrds_csq_pit
        WHERE indfmt = 'INDL'
        AND datafmt = 'STD'
        AND consol = 'C'
        AND popsrc = 'D'
        AND curcdq = 'USD'
    )
    SELECT DISTINCT
        s.permno,
        l.lpermno,
        l.lpermco,
        l.gvkey,
        l.liid as iid,
    FROM ccmxpf_lnkhist l
    INNER JOIN valid_gvkeys v ON v.gvkey = l.gvkey
    INNER JOIN stkdlysecuritydata s ON s.permno = l.lpermno
        AND s.dlycaldt = DATE '{date}'
    WHERE 
        l.linktype IN ('LC', 'LU')
        AND l.linkprim IN ('P', 'C')
        AND l.linkdt <= DATE '{date}'
        AND (l.linkenddt >= DATE '{date}' OR l.linkenddt IS NULL)
        AND l.gvkey IS NOT NULL
    """
    
    return duck_conn.execute(universe_query.format(date=date_str)).fetchdf()


def process_factors(start_date, end_date, output_base_folder, factors):
    """
    Main processing function to calculate multiple factors over a date range.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_base_folder (str): Base folder for storing factor data
        factors (list): List of FactorCalculator instances
    """
    with duckdb.connect('wrds_data.db', read_only=True) as duck_conn:
        trading_dates = get_trading_dates(start_date, end_date, duck_conn)
        print(f"Processing {len(trading_dates)} trading dates")
        
        for date in trading_dates:
            date_str = date.strftime('%Y-%m-%d')
            print(f"\nProcessing date: {date_str}")
            
            try:
                # Get universe for this date
                universe_df = get_all_investment_universe(date_str, duck_conn)
                print(f"Universe size: {len(universe_df)} stocks")
                
                # Calculate each factor
                for factor_calculator in factors:
                    try:
                        factor_df = factor_calculator.calculate(date_str, universe_df, duck_conn)
                        
                        # Save to parquet
                        output_folder = factor_calculator.get_output_folder(output_base_folder)
                        output_file = f"{output_folder}/{factor_calculator.factor_name}_{date_str}.parquet"
                        factor_df.to_parquet(output_file)
                        
                    except Exception as e:
                        print(f"Error calculating {factor_calculator.factor_name} for {date_str}: {str(e)}")
                        continue
                    
            except Exception as e:
                print(f"Error processing date {date_str}: {str(e)}")
                continue



'''
def process_single_date(date, output_base_folder, factors, db_path='wrds_data.db'):
    """
    Process a single date for all factors.
    
    This function is designed to be called by a thread pool.
    
    Args:
        date: The date to process
        output_base_folder: Base folder for storing factor data
        factors: List of FactorCalculator instances
        db_path: Path to the DuckDB database
    
    Returns:
        tuple: (date_str, status, message)
    """
    date_str = date.strftime('%Y-%m-%d')
    
    try:
        # Create a new database connection for this thread
        with duckdb.connect(db_path, read_only=True) as duck_conn:
            # Get universe for this date
            universe_df = get_all_investment_universe(date_str, duck_conn)
            
            # Calculate each factor
            results = []
            for factor_calculator in factors:
                try:
                    factor_df = factor_calculator.calculate(date_str, universe_df, duck_conn)
                    
                    # Save to parquet
                    output_folder = factor_calculator.get_output_folder(output_base_folder)
                    Path(output_folder).mkdir(parents=True, exist_ok=True)
                    output_file = f"{output_folder}/{factor_calculator.factor_name}_{date_str}.parquet"
                    factor_df.to_parquet(output_file)
                    
                    results.append((factor_calculator.factor_name, "success"))
                except Exception as e:
                    results.append((factor_calculator.factor_name, f"error: {str(e)}"))
            
            return date_str, "success", {"universe_size": len(universe_df), "factors": results}
    except Exception as e:
        return date_str, "error", str(e)

def process_factors(start_date, end_date, output_base_folder, factors, max_workers=None):
    """
    Main processing function to calculate multiple factors over a date range using multi-threading.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_base_folder (str): Base folder for storing factor data
        factors (list): List of FactorCalculator instances
        max_workers (int, optional): Maximum number of worker threads. If None, uses default from ThreadPoolExecutor.
    """
    # If max_workers is not specified, the default will be min(32, os.cpu_count() + 4)
    if max_workers is None:
        import os
        max_workers = min(32, os.cpu_count() + 4)
    
    with duckdb.connect('wrds_data.db', read_only=True) as duck_conn:
        trading_dates = get_trading_dates(start_date, end_date, duck_conn)
        print(f"Processing {len(trading_dates)} trading dates using {max_workers} worker threads")
    
    # Create output base directory
    Path(output_base_folder).mkdir(parents=True, exist_ok=True)
    
    # Use ThreadPoolExecutor to parallelize date processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all date processing tasks
        future_to_date = {
            executor.submit(process_single_date, date, output_base_folder, factors): date 
            for date in trading_dates
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_date), total=len(trading_dates), desc="Processing dates"):
            date = future_to_date[future]
            date_str = date.strftime('%Y-%m-%d')
            
            try:
                result = future.result()
                status = result[1]
                
                if status == "success":
                    universe_size = result[2]["universe_size"]
                    print(f"\nDate {date_str}: Processed {universe_size} stocks successfully")
                    
                    # Optionally show detailed factor results
                    # for factor_result in result[2]["factors"]:
                    #     print(f"  - {factor_result[0]}: {factor_result[1]}")
                else:
                    print(f"\nError processing date {date_str}: {result[2]}")
            
            except Exception as e:
                print(f"\nException while processing date {date_str}: {str(e)}")
'''

def process_neutralized_factors(start_date, end_date, input_base_folder, output_base_folder, 
                               factor_names, neutralization_types=None):
    """
    Process existing factors and create neutralized versions.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        input_base_folder (str): Base folder where original factors are stored
        output_base_folder (str): Base folder for storing neutralized factor data
        factor_names (list): List of factor names to neutralize
        neutralization_types (list): List of neutralization types. Default is ['naics', 'size', 'combined']
    """
    if neutralization_types is None:
        neutralization_types = ['naics', 'size', 'combined']
    
    with duckdb.connect('wrds_data.db', read_only=True) as duck_conn:
        trading_dates = get_trading_dates(start_date, end_date, duck_conn)
        print(f"Processing neutralized factors for {len(trading_dates)} trading dates")
        
        # Create neutralizer
        neutralizer = FactorNeutralizer()
        
        # Create output folders
        os.makedirs(output_base_folder, exist_ok=True)
        
        # Process each date
        for date in tqdm(trading_dates, desc="Processing dates"):
            date_str = date.strftime('%Y-%m-%d')
            parquet_date_str = pd.to_datetime(date_str).strftime('%Y%m%d')
            
            #print(f"\nProcessing date: {date_str}")
            
            # Process each factor
            for factor_name in factor_names:
                try:
                    # Load original factor data
                    input_file = f"{input_base_folder}/{factor_name}/{factor_name}_{date_str}.parquet"
                    if not os.path.exists(input_file):
                        print(f"Factor file not found: {input_file}")
                        continue
                        
                    # Load base data (contains NAICS and market cap)
                    base_data_file = f"Data_all/Base_Data/{parquet_date_str}.parquet"
                    
                    if not os.path.exists(base_data_file):
                        print(f"Base data file not found: {base_data_file}")
                        continue
                    
                    # Load factor and base data
                    factor_df = pd.read_parquet(input_file)
                    base_df = pd.read_parquet(base_data_file)
                    
                    # Always use the last column as the factor value
                    last_col = factor_df.columns[-1]
                    
                    # Create a copy with the factor column renamed to the expected name
                    factor_df = factor_df.copy()
                    factor_df[factor_name] = factor_df[last_col]
                    
                    # Merge to get NAICS and market cap data with factor data
                    if 'permno' in factor_df.columns:
                        merged_df = factor_df.merge(
                            base_df[['naics', 'naics_sector', 'dlycap']], 
                            left_on='permno', 
                            right_index=True,
                            how='left'
                        )
                    else:
                        # Assuming factor_df is indexed by permno
                        merged_df = factor_df.merge(
                            base_df[['naics', 'naics_sector', 'dlycap']],
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                    
                    # Apply each type of neutralization
                    for neut_type in neutralization_types:
                        try:
                            if neut_type == 'naics':
                                neutralized_df = neutralizer.neutralize_by_naics(
                                    merged_df, 
                                    factor_name,
                                    winsorize_factor=True,
                                    standardize_factor=True
                                )
                                output_suffix = 'naics_neutral'
                            elif neut_type == 'size':
                                neutralized_df = neutralizer.neutralize_by_size(
                                    merged_df, 
                                    factor_name,
                                    winsorize_factor=True,
                                    standardize_factor=True
                                )
                                output_suffix = 'size_neutral'
                            elif neut_type == 'combined':
                                neutralized_df = neutralizer.neutralize_combined(
                                    merged_df, 
                                    factor_name,
                                    winsorize_factor=True,
                                    standardize_factor=True
                                )
                                output_suffix = 'neutral'
                            else:
                                print(f"Unknown neutralization type: {neut_type}")
                                continue
                            
                            # Create output folder
                            output_folder = f"{output_base_folder}/{factor_name}_{output_suffix}"
                            os.makedirs(output_folder, exist_ok=True)
                            
                            # Prepare output file
                            output_file = f"{output_folder}/{factor_name}_{output_suffix}_{date_str}.parquet"
                            
                            # Extract output column name
                            output_col = f"{factor_name}_{output_suffix}"
                            
                            # Check if output column exists
                            if output_col not in neutralized_df.columns:
                                print(f"Output column '{output_col}' not found in neutralized dataframe")
                                continue
                            
                            # Keep only necessary columns
                            id_cols = ['permno', 'date', 'gvkey', 'iid', 'lpermno', 'lpermco']
                            cols_to_keep = [col for col in id_cols if col in neutralized_df.columns]
                            cols_to_keep.append(output_col)
                            
                            # Save neutralized factor
                            neutralized_df[cols_to_keep].to_parquet(output_file)
                            #print(f"Saved {output_suffix} {factor_name} to {output_file}")
                        
                        except Exception as e:
                            import traceback
                            print(f"Error with {neut_type} neutralization for {factor_name}: {str(e)}")
                            print(traceback.format_exc())
                            continue
                    
                except Exception as e:
                    import traceback
                    print(f"Error processing {factor_name} for {date_str}: {str(e)}")
                    print(traceback.format_exc())
                    continue


def examine_factor_file(date_str, factor_name, base_folder="factor_data"):
    """
    Read and analyze factor data for a specific date. Provides detailed analysis 
    of factor values, data quality, and distribution characteristics.
    
    Args:
        date_str (str): Date in 'YYYY-MM-DD' format to analyze
        factor_name (str): Name of the factor (e.g., 'roe')
        base_folder (str): Base path to the factor data folders
        
    Returns:
        pandas.DataFrame: The loaded factor data if successful, None otherwise
    """
    # Convert factor_name to lowercase to match file naming convention
    factor_name = factor_name.lower()
    # Construct the full file path
    file_path = f"{base_folder}/{factor_name}/{factor_name}_{date_str}.parquet"
    print(file_path)
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Basic information about the dataset
        print(f"\nAnalysis of {factor_name.upper()} data for {date_str}")
        print("=" * 50)
        print("\nBasic Information:")
        print(f"Total number of stocks: {len(df)}")
        print(f"Number of stocks with {factor_name}: {df[factor_name].notna().sum()}")
        print(f"Coverage ratio: {(df[factor_name].notna().sum() / len(df)) * 100:.2f}%")
        
        # Factor value statistics
        print("\nFactor Statistics:")
        stats = df[factor_name].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
        print(stats)
        
        # Distribution analysis part of examine_factor_file function
        print("\nDistribution Analysis:")

        # Calculate quantiles for non-null values
        factor_values = df[factor_name].dropna()

        # Create quantile categories
        quantiles = pd.qcut(factor_values, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Calculate statistics for each quantile, explicitly setting observed=True
        quantile_stats = pd.DataFrame({
            'count': quantiles.value_counts(),
            'min': factor_values.groupby(quantiles, observed=True).min(),
            'max': factor_values.groupby(quantiles, observed=True).max(),
            'mean': factor_values.groupby(quantiles, observed=True).mean()
        }).round(4)

        print("\nQuantile breakdown:")
        print(quantile_stats)
        
        # Extreme values analysis
        print("\nExtreme Values:")
        print("\nTop 5 highest values:")
        top_5 = df.nlargest(5, factor_name)[[factor_name, 'gvkey', 'iid']]
        print(top_5)
        
        print("\nBottom 5 lowest values:")
        bottom_5 = df.nsmallest(5, factor_name)[[factor_name, 'gvkey', 'iid']]
        print(bottom_5)
        
        # Missing value analysis
        print("\nMissing Value Analysis:")
        missing_analysis = pd.DataFrame({
            'missing_count': df.isna().sum(),
            'missing_percentage': (df.isna().sum() / len(df) * 100).round(2)
        })
        print(missing_analysis)
        
        # Identifier coverage
        print("\nIdentifier Coverage:")
        id_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']
        for col in id_columns:
            unique_count = df[col].nunique()
            print(f"Unique {col}: {unique_count}")
        
        return df
        
    except FileNotFoundError:
        print(f"No {factor_name} data file found for {date_str}")
        return None
    except Exception as e:
        print(f"Error analyzing {factor_name} data for {date_str}: {str(e)}")
        return None

# Example of how to use the function for different types of analysis
def analyze_factor_over_time(factor_name, start_date, end_date, base_folder="factor_data"):
    """
    Analyze a factor's behavior over a time period, showing trends and stability.
    
    Args:
        factor_name (str): Name of the factor to analyze
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        base_folder (str): Base path to factor data
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start, end, freq='B')  # Business days
    
    # Storage for time series statistics
    time_series_stats = []
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        df = examine_factor_file(date_str, factor_name, base_folder)
        
        if df is not None:
            stats = {
                'date': date,
                'coverage': df[factor_name].notna().sum() / len(df),
                'mean': df[factor_name].mean(),
                'median': df[factor_name].median(),
                'std': df[factor_name].std()
            }
            time_series_stats.append(stats)
    
    if time_series_stats:
        stats_df = pd.DataFrame(time_series_stats)
        print("\nTime Series Analysis:")
        print(stats_df.describe())
        
        return stats_df
    
    return None


def get_trading_dates(start_date, end_date, db_conn):
    """
    Get list of trading dates from CRSP calendar.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    db_conn : duckdb.DuckDBPyConnection
        Active connection to DuckDB database
    
    Returns:
    --------
    list : Trading dates in YYYY-MM-DD format
    """
    trading_dates_query = f"""
        SELECT DISTINCT caldt as trading_date
        FROM metaexchangecalendar
        WHERE tradingflg = 'Y'
        AND caldt BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ORDER BY caldt
    """
    return db_conn.execute(
        trading_dates_query.format(start_date=start_date, end_date=end_date)
    ).fetchdf()['trading_date'].tolist()

'''
def sql_base_data(date, parquet_path,db_conn):
    """
    Load and cache basic market data for a single date.
    Uses dlycaldt as the date column which is specific to dsf_v2 table.
    
    Parameters:
    -----------
    date : str
        Date in 'YYYY-MM-DD' format
    parquet_path : str
        Path template for saving parquet files
        
    Returns:
    --------
    DataFrame : Daily stock data with standardized formatting
    """
    query = f"""
        SELECT 
            s.permno,
            s.dlycaldt as date,
            s.dlyprc,
            s.dlyprevprc,
            s.dlyret,
            s.dlyretmissflg,
            s.dlycap,
            s.tradingstatusflg,
            s.securitytype,
            s.sharetype,
            s.dlyclose,
            s.dlyopen
        FROM dsf_v2 s
        WHERE s.dlycaldt = DATE '{date}'
        ORDER BY s.dlycaldt
    """
    
    # Execute query directly with f-string - no need for .format()
    base_info_df = db_conn.execute(query).fetchdf()
    

    base_info_df.set_index('permno', inplace=True)
    
    # Convert date to YYYYMMDD format for parquet filename
    parquet_date = pd.to_datetime(date).strftime('%Y%m%d')
    
    if len(base_info_df) > 0:
        # Use the formatted date in the parquet path
        base_info_df.to_parquet(parquet_path.format(parquet_date))
    
    return base_info_df

'''

def sql_base_data(date, parquet_path, db_conn):
    """
    Load and cache basic market data for a single date, including NAICS industry codes.
    Uses dlycaldt as the date column which is specific to dsf_v2 table.
    
    Parameters:
    -----------
    date : str
        Date in 'YYYY-MM-DD' format
    parquet_path : str
        Path template for saving parquet files
    db_conn : duckdb.DuckDBPyConnection
        Connection to the database
        
    Returns:
    --------
    DataFrame : Daily stock data with standardized formatting and NAICS industry codes
    """
    # First, get the basic stock data
    query = f"""
        SELECT 
            s.permno,
            s.dlycaldt as date,
            s.dlyprc,
            s.dlyprevprc,
            s.dlyret,
            s.dlyretmissflg,
            s.dlycap,
            s.tradingstatusflg,
            s.securitytype,
            s.sharetype,
            s.dlyclose,
            s.dlyopen
        FROM dsf_v2 s
        WHERE s.dlycaldt = DATE '{date}'
        ORDER BY s.dlycaldt
    """
    
    # Execute query directly
    base_info_df = db_conn.execute(query).fetchdf()
    
    # Now get NAICS data for all permnos
    permnos = base_info_df['permno'].tolist()
    
    if permnos:
        naics_query = f"""
        WITH naics_data AS (
            SELECT 
                s.permno,
                COALESCE(s.naics, '0') as naics,
                CASE 
                    WHEN s.naics IS NULL OR s.naics = '' THEN '0'
                    ELSE SUBSTRING(s.naics, 1, 2) 
                END as naics_sector,
                s.secinfostartdt,
                s.secinfoenddt
            FROM stksecurityinfohist s
            WHERE s.permno IN ({', '.join([str(p) for p in permnos])})
            AND s.secinfostartdt <= DATE '{date}'
            AND (s.secinfoenddt >= DATE '{date}' OR s.secinfoenddt IS NULL)
        )
        SELECT 
            permno,
            naics,
            naics_sector
        FROM naics_data
        """
        
        try:
            naics_df = db_conn.execute(naics_query).fetchdf()
            
            # Join the NAICS data to our base info
            base_info_df = base_info_df.merge(
                naics_df,
                on='permno',
                how='left'
            )
            
            # Fill missing NAICS codes with '0'
            base_info_df['naics'] = base_info_df['naics'].fillna('0')
            base_info_df['naics_sector'] = base_info_df['naics_sector'].fillna('0')
            
        except Exception as e:
            print(f"Error retrieving NAICS data: {str(e)}")
            # Still continue with the base data
            base_info_df['naics'] = '0'
            base_info_df['naics_sector'] = '0'
    
    # Set index to permno
    base_info_df.set_index('permno', inplace=True)
    
    # Convert date to YYYYMMDD format for parquet filename
    parquet_date = pd.to_datetime(date).strftime('%Y%m%d')
    
    if len(base_info_df) > 0:
        # Use the formatted date in the parquet path
        base_info_df.to_parquet(parquet_path.format(parquet_date))
    
    return base_info_df


def load_factor_data(date, factor_names, factor_data_path='factor_data'):
    """
    Load multiple factors' data from parquet files.
    
    Parameters:
    -----------
    date : str
        Date in 'YYYY-MM-DD' format
    factor_names : list
        List of factor names
    factor_data_path : str
        Base directory for factor data
        
    Returns:
    --------
    DataFrame : Combined factor data for the date
    """
    date_formatted = pd.to_datetime(date).strftime('%Y-%m-%d')
    
    # Initialize with None
    combined_df = None
    
    for factor_name in factor_names:
        # Construct path for each factor
        factor_file = Path(factor_data_path) / str(factor_name) / f"{factor_name}_{date_formatted}.parquet"
        
        if not factor_file.exists():
            print(f"File not found: {factor_file}")
            continue
            
        # Load the current factor data
        curr_df = pd.read_parquet(factor_file)
        
        # Skip if permno column is missing
        if 'permno' not in curr_df.columns:
            print(f"Warning: permno column missing in {factor_name} data for {date_formatted}")
            continue
        
        # For the first valid factor, use it as our base
        if combined_df is None:
            # Keep this DataFrame as is - with row numbers as index
            combined_df = curr_df
        else:
            # For additional factors, keep only permno and the factor column
            factor_cols = ['permno']
            if factor_name in curr_df.columns:
                factor_cols.append(factor_name)
            
            # Only keep essential columns
            curr_df = curr_df[factor_cols]
            
            # Merge on permno column (not index)
            combined_df = pd.merge(
                combined_df,
                curr_df,
                on='permno',
                how='outer'  # Include all stocks from both datasets
            )
    
    return combined_df


def sql_extended_base_data(date, parquet_path, db_conn):
    """
    Load and cache extended market data for backtesting.
    Includes additional fields for corporate actions and trading.
    
    Parameters:
    -----------
    date : str
        Date in 'YYYY-MM-DD' format
    parquet_path : str
        Path template for saving parquet files
    db_conn : duckdb.DuckDBPyConnection
        Database connection
        
    Returns:
    --------
    DataFrame : Extended daily stock data
    """
    query = f"""
        SELECT 
            s.permno,
            s.ticker,
            s.dlycaldt as date,
            s.dlyprc,
            s.dlyprevprc,
            s.dlyret,
            s.dlyretmissflg,
            s.dlycap,
            s.tradingstatusflg,
            s.securitytype,
            s.sharetype,
            s.dlyclose,
            s.dlyopen,
            s.dlyprcflg,
            s.dlyprevprcflg,
            s.dlydelflg,
            s.dlydistretflg,
            s.dlyfacprc,
            s.dlyclose,
            s.dlyopen
        FROM dsf_v2 s
        WHERE s.dlycaldt = DATE '{date}'
        ORDER BY s.dlycaldt
    """
    
    # Execute query
    extended_info_df = db_conn.execute(query).fetchdf()
    
    # Set index
    extended_info_df.set_index('permno', inplace=True)
    
    # Convert date to YYYYMMDD format for parquet filename
    parquet_date = pd.to_datetime(date).strftime('%Y%m%d')
    
    if len(extended_info_df) > 0:
        # Save to extended data directory
        extended_path = parquet_path.format(parquet_date)
        
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(extended_path), exist_ok=True)
        
        # Save extended data
        extended_info_df.to_parquet(extended_path)
    
    return extended_info_df

def create_analysis_data(start_date, end_date, factor_names, db_path, factor_data_path):
    """
    Create dataset for factor analysis with multiple factors.
    """
    if isinstance(factor_names, str):
        factor_names = [factor_names]  # Convert single factor to list
        
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

        cache_dir = os.path.join('Data_all', 'Base_Data')
        parquet_path = os.path.join(cache_dir, '{}.parquet')
        os.makedirs(cache_dir, exist_ok=True)

        data_df_dic = {}
        
        for date in tqdm(trading_dates, desc="Processing dates"):
            try:
                date_str = pd.to_datetime(date).strftime('%Y%m%d')
                
                # Load base data
                if os.path.exists(parquet_path.format(date_str)):
                    base_df = pd.read_parquet(parquet_path.format(date_str))
                else:
                    base_df = sql_base_data(date, parquet_path, db_conn)

                # Load all factors
                factor_df = load_factor_data(date, factor_names, factor_data_path)
                
                if factor_df is not None and 'permno' in factor_df.columns:
                    # Reset index on base_df to get permno as a column
                    base_df_reset = base_df.reset_index()
                    
                    # Remove duplicate columns (except permno)
                    duplicate_cols = base_df_reset.columns.intersection(factor_df.columns).difference(['permno'])
                    if not duplicate_cols.empty:
                        factor_df = factor_df.drop(columns=duplicate_cols)
                    
                    # Merge on permno column
                    merged_df = pd.merge(
                        base_df_reset,  # permno is now a column
                        factor_df,      # already has permno as column
                        on='permno',
                        how='left'      # keep all stocks in base_df
                    )
                    
                    # Set permno back as index
                    merged_df.set_index('permno', inplace=True)
                    
                    # Store the properly merged dataframe
                    data_df_dic[date] = merged_df
                else:
                    # If no factor data, just use the base data
                    data_df_dic[date] = base_df
                        
            except Exception as e:
                print(f"\nError processing date {date}: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        return data_df_dic, trading_dates

    finally:
        db_conn.close()



def create_extended_analysis_data(start_date, end_date, factor_names, db_path, factor_data_path):
    """
    Create dataset for backtesting with extended data.
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
        
        trading_dates = get_trading_dates(prev_start_date, end_date, db_conn)

        # Use Extended_Base_Data directory
        cache_dir = os.path.join('Data_all', 'Extended_Base_Data')
        parquet_path = os.path.join(cache_dir, '{}.parquet')
        os.makedirs(cache_dir, exist_ok=True)

        data_df_dic = {}
        
        for date in tqdm(trading_dates, desc="Processing dates"):
            try:
                date_str = pd.to_datetime(date).strftime('%Y%m%d')
                
                # Load extended base data
                if os.path.exists(parquet_path.format(date_str)):
                    base_df = pd.read_parquet(parquet_path.format(date_str))
                else:
                    base_df = sql_extended_base_data(date, parquet_path, db_conn)

                # Load factors
                factor_df = load_factor_data(date, factor_names, factor_data_path)
                
                if factor_df is not None:
                    # The base_df has permno as index
                    # First reset the index to make permno a column
                    base_df_with_permno = base_df.reset_index()
                    
                    # Merge on permno column
                    merged_df = pd.merge(
                        base_df_with_permno,
                        factor_df,
                        on='permno',
                        how='left'  # Keep all stocks in base_df
                    )
                    
                    # Set permno back as index if needed
                    merged_df.set_index('permno', inplace=True)
                    
                    # Store result
                    data_df_dic[date] = merged_df
                else:
                    # Just store the base data
                    data_df_dic[date] = base_df
                        
            except Exception as e:
                print(f"\nError processing date {date}: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        return data_df_dic, trading_dates

    finally:
        db_conn.close()


        

def create_benchmark_data(db_conn, gvkeyx, start_date, end_date, output_path='Data_all/Benchmark_data'):
    """
    Extract benchmark data and save to parquet.
    
    Parameters:
    -----------
    db_conn : duckdb.DuckDBPyConnection
        Database connection
    gvkeyx : str
        Global Index Key (e.g., '000003' for S&P 500)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    output_path : str
        Base path to save benchmark data
    """
    # Create output directory for specific benchmark
    benchmark_dir = os.path.join(output_path, gvkeyx)
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Query benchmark data
    query = f"""
        SELECT 
            datadate as date,
            gvkeyx,
            prccddiv as tr_idx,
            prccd as close_idx,
            prchd as high_idx,
            prcld as low_idx,
            dvpsxd as div_idx
        FROM idx_daily
        WHERE gvkeyx = '{gvkeyx}'
        AND datadate BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ORDER BY datadate
    """
    
    benchmark_df = db_conn.execute(query).fetchdf()
    
    # Calculate daily returns
    benchmark_df['ret_idx'] = benchmark_df['tr_idx'].pct_change()
    
    # Save to parquet
    output_file = os.path.join(benchmark_dir, f'benchmark_{gvkeyx}.parquet')
    benchmark_df.to_parquet(output_file)
    
    print(f"Saved benchmark data for {gvkeyx} to {output_file}")
    print(f"Date range: {benchmark_df['date'].min()} to {benchmark_df['date'].max()}")
    print(f"Number of records: {len(benchmark_df)}")
    
    return benchmark_df

def process_benchmarks(conn, start_date, end_date, benchmark_list=None):
    """
    Process multiple benchmarks and save their data.
    
    Parameters:
    -----------
    db_path : str
        Path to the database file
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    benchmark_list : list
        List of benchmark gvkeyx to process. If None, defaults to ['000003']
    """
    if benchmark_list is None:
        benchmark_list = ['000003']  # Default to S&P 500
        
    # Connect to database
   
    for gvkeyx in tqdm(benchmark_list, desc="Processing benchmarks"):
        try:
            create_benchmark_data(conn, gvkeyx, start_date, end_date)
        except Exception as e:
            print(f"Error processing benchmark {gvkeyx}: {str(e)}")
            continue


