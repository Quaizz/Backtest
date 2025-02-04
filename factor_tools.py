# factor_tools.py

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime



def get_trading_dates(start_date, end_date, duck_conn):
    """
    Retrieve all trading dates between start_date and end_date from CRSP data.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        duck_conn: Active DuckDB connection
        
    Returns:
        list: List of trading dates in chronological order
    """
    dates_query = """
    SELECT DISTINCT dlycaldt as trading_date
    FROM stkdlysecuritydata
    WHERE dlycaldt BETWEEN DATE '{start_date}' AND DATE '{end_date}'
    ORDER BY dlycaldt
    """
    
    return duck_conn.execute(
        dates_query.format(start_date=start_date, end_date=end_date)
    ).fetchdf()['trading_date'].tolist()

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