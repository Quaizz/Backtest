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
def create_analysis_data(start_date, end_date, factor_name, db_path, factor_data_path):
    """
    Create dataset for factor analysis, following the efficient patterns
    of the Chinese version but adapted for US market needs.
    """
    db_conn = duckdb.connect(db_path, read_only=True)
    
    try:
        trading_dates = get_trading_dates(start_date, end_date, db_conn)
        
        cache_dir = os.path.join('Data_all', 'Base_Data')
        parquet_path = os.path.join(cache_dir, '{}.parquet')
        os.makedirs(cache_dir, exist_ok=True)
        
        data_df_dic = {}
        for date in trading_dates:
            if os.path.exists(parquet_path.format(date)):
                base_df = pd.read_parquet(parquet_path.format(date))
            else:
                base_df = sql_base_data(date, parquet_path,db_conn)
            
            factor_df = load_factor_data(date, factor_name, factor_data_path)
            # Set 'permno' as index for factor DataFrame
            factor_df.set_index('permno', inplace=True)

            if factor_df is not None:
                data_df_dic[date] = pd.concat([base_df, factor_df], axis=1)
        
        return data_df_dic, trading_dates
        
    finally:
        db_conn.close()
'''
def create_analysis_data(start_date, end_date, factor_name, db_path, factor_data_path):
    """
    Create dataset for factor analysis with parallel processing.
    Loads all stocks without filtering to allow flexibility in backtesting.
    """
    db_conn = duckdb.connect(db_path, read_only=True)
    
    try:
        # Get previous trading date before start_date
        prev_date_query = f"""
            SELECT MAX(caldt) as prev_date
            FROM metaexchangecalendar
            WHERE tradingflg = 'Y'
            And caldt < DATE '{start_date}'
        """

        prev_start_date = db_conn.execute(
            prev_date_query, 
        ).fetchone()[0]
        
        trading_dates = get_trading_dates(prev_start_date, end_date, db_conn)

    

        cache_dir = os.path.join('Data_all', 'Base_Data')
        parquet_path = os.path.join(cache_dir, '{}.parquet')
        os.makedirs(cache_dir, exist_ok=True)

        data_df_dic = {}
        
        # Process each date sequentially with detailed logging
        for date in tqdm(trading_dates, desc="Processing dates"):
            try:
                #print(f"\nProcessing date: {date}")
                date_str = pd.to_datetime(date).strftime('%Y%m%d')
                if os.path.exists(parquet_path.format(date_str)):
                    #print(f"Loading cached data for {date}")
                    
                    #print(parquet_path.format(date))
                    base_df = pd.read_parquet(parquet_path.format(date_str))
                else:
                    #print(f"Getting base data for {date}")
                    
                    #print(parquet_path.format(date))
                    base_df = sql_base_data(date, parquet_path, db_conn)

                #print(f"Loading factor data for {date}")
                factor_df = load_factor_data(date, factor_name, factor_data_path)
                
                if factor_df is not None:
                    #print(f"Combining data for {date}")
                    factor_df.set_index('permno', inplace=True)
                    data_df_dic[date] = pd.concat([base_df, factor_df], axis=1)
                else:
                    print(f"No factor data found for {date}")
                        
            except Exception as e:
                print(f"\nError processing date {date}: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(traceback.format_exc())
                continue



        return data_df_dic, trading_dates

    finally:
        db_conn.close()




def load_factor_data(date, factor_name, factor_data_path='factor_data'):
    """
    Load pre-calculated factor data from parquet file.
    
    Parameters:
    -----------
    date : str
        Date in 'YYYY-MM-DD' format
    factor_name : str
        Name of the factor
    base_path : str
        Base directory for factor data
        
    Returns:
    --------
    DataFrame : Factor data for the date
    """

    # Convert date string to datetime and format it as YYYYMMDD
    date_formatted = pd.to_datetime(date).strftime('%Y-%m-%d')

    factor_file = Path(factor_data_path) / factor_name / f"{factor_name}_{date_formatted}.parquet"
    
    if not factor_file.exists():
        print(f"File not found: {factor_file}")
        return None
        
    return pd.read_parquet(factor_file)



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


