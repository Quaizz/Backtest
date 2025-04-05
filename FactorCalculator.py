
# FactorCalculator.py

import pandas as pd
import os
from pathlib import Path


# factor_neutralization.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
import warnings
import os

class FactorNeutralizer:
    """
    Class for performing factor neutralization to remove unwanted factor exposures.
    
    This class provides methods to neutralize factors with respect to:
    - Industry/Sector exposures (NAICS codes)
    - Size (market cap)
    - Any combination of the above
    
    Includes preprocessing steps like winsorization and standardization.
    """
    def __init__(self):
        """Initialize the FactorNeutralizer."""
        pass
    
    def _winsorize(self, factor, n=3):
        """
        Winsorize a factor series using median absolute deviation (MAD).
        
        Parameters:
        -----------
        factor : Series
            Factor values
        n : int, default 3
            Number of MADs to use for winsorization
            
        Returns:
        --------
        Series : Winsorized factor values
        """
        r = factor.dropna().copy()
        med = np.median(r)
        ximed = abs(r - med)
        mad = np.median(ximed)
        r[r < (med-n*mad)] = med - n*mad
        r[r > (med+n*mad)] = med + n*mad
        return r

    def _standardize(self, factor, method=2):
        """
        Standardize a factor series.
        
        Parameters:
        -----------
        factor : Series
            Factor values
        method : int, default 2
            Standardization method:
            1 = Min-max scaling (0 to 1)
            2 = Z-score standardization (mean 0, std 1)
            3 = Scale by powers of 10
            
        Returns:
        --------
        Series : Standardized factor values
        """
        data = factor.dropna().copy()
        
        if method == 1:
            # Min-max scaling
            return (data - data.min()) / (data.max() - data.min())
        elif method == 2:
            # Z-score standardization
            return (data - data.mean()) / data.std()
        elif method == 3:
            # Scale by powers of 10
            return data / 10**np.ceil(np.log10(data.abs().max()))
        else:
            return data
    
    def _create_dummy_variables(self, df, column):
        """
        Create dummy variables for categorical column.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing the categorical column
        column : str
            Name of the categorical column
            
        Returns:
        --------
        DataFrame
            DataFrame with dummy variables added
        """
        # Handle missing values
        df[column] = df[column].fillna('0')
        
        # Create dummy variables
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
        
        # Combine with original DataFrame
        return pd.concat([df, dummies], axis=1)
    
    def neutralize_by_naics(self, factor_df, factor_name, naics_column='naics_sector', 
                          winsorize_factor=True, standardize_factor=True, mad_n=3):
        """
        Neutralize factor with respect to NAICS industry exposures.
        
        Parameters:
        -----------
        factor_df : DataFrame
            DataFrame containing the factor and NAICS information
        factor_name : str
            Name of the factor column to be neutralized
        naics_column : str, default 'naics_sector'
            Name of the NAICS column (typically 'naics_sector' for 2-digit NAICS codes)
        winsorize_factor : bool, default True
            Whether to winsorize the factor before neutralization
        standardize_factor : bool, default True
            Whether to standardize the factor before neutralization
        mad_n : int, default 3
            Number of MADs to use for winsorization
            
        Returns:
        --------
        DataFrame
            DataFrame with the neutralized factor added as a new column
        """
        if naics_column not in factor_df.columns:
            warnings.warn(f"NAICS column '{naics_column}' not found in DataFrame. Returning original factor.")
            return factor_df
        
        # Create copy to avoid modifying the original
        df = factor_df.copy()
        
        # Handle missing factor values
        df = df.dropna(subset=[factor_name])
        
        # Apply winsorization and standardization if requested
        if winsorize_factor:
            y_raw = df[factor_name]
            y = self._winsorize(y_raw, n=mad_n)
        else:
            y = df[factor_name]
        
        if standardize_factor:
            y = self._standardize(y)
        
        # Store processed y values back in the dataframe
        df[f'{factor_name}_processed'] = y
        
        # Create NAICS dummy variables
        df = self._create_dummy_variables(df, naics_column)
        
        # Get NAICS dummy columns
        naics_dummies = [col for col in df.columns if col.startswith(f"{naics_column}_")]
        
        if not naics_dummies:
            warnings.warn("No NAICS dummy variables created. Returning original factor.")
            return factor_df
        
        try:
            # Perform regression of factor on NAICS dummies
            X = df[naics_dummies]
            X = sm.add_constant(X)  # Add constant
            
            # Use processed y values
            y = df[f'{factor_name}_processed']
            
            model = OLS(y, X).fit()
            
            # Calculate residuals (NAICS-neutralized factor)
            df[f'{factor_name}_naics_neutral'] = model.resid
            
            # Standardize the neutralized factor
            df[f'{factor_name}_naics_neutral'] = self._standardize(df[f'{factor_name}_naics_neutral'])
            
            # Return original DataFrame with neutralized factor added
            result = factor_df.copy()
            result = result.merge(
                df[[f'{factor_name}_naics_neutral']],
                left_index=True,
                right_index=True,
                how='left'
            )
            
            return result
            
        except Exception as e:
            warnings.warn(f"Error in NAICS neutralization: {str(e)}")
            return factor_df
    
    def neutralize_by_size(self, factor_df, factor_name, size_column='dlycap', 
                     winsorize_factor=True, standardize_factor=True, mad_n=3):
        """
        Neutralize factor with respect to market capitalization (size).
        
        Parameters:
        -----------
        factor_df : DataFrame
            DataFrame containing the factor and market cap information
        factor_name : str
            Name of the factor column to be neutralized
        size_column : str, default 'dlycap'
            Name of the market cap column
        winsorize_factor : bool, default True
            Whether to winsorize the factor before neutralization
        standardize_factor : bool, default True
            Whether to standardize the factor before neutralization
        mad_n : int, default 3
            Number of MADs to use for winsorization
            
        Returns:
        --------
        DataFrame
            DataFrame with the size-neutralized factor added as a new column
        """
        if size_column not in factor_df.columns:
            warnings.warn(f"Size column '{size_column}' not found in DataFrame. Returning original factor.")
            return factor_df
        
        # Create copy to avoid modifying the original
        df = factor_df.copy()
        
        # Handle missing factor values
        df = df.dropna(subset=[factor_name])
        
        # Handle missing or zero market cap values by replacing with mean
        valid_sizes = df[df[size_column] > 0][size_column]
        if len(valid_sizes) > 0:
            mean_size = valid_sizes.mean()
            df[size_column] = df[size_column].replace(0, np.nan).fillna(mean_size)
        else:
            warnings.warn("No valid size values found. Size neutralization may not be effective.")
        
        # Apply winsorization and standardization if requested
        if winsorize_factor:
            y_raw = df[factor_name]
            y = self._winsorize(y_raw, n=mad_n)
        else:
            y = df[factor_name]
        
        if standardize_factor:
            y = self._standardize(y)
        
        # Store processed y values back in the dataframe
        df[f'{factor_name}_processed'] = y
        
        # Log transform market cap to normalize its distribution
        df['log_' + size_column] = np.log(df[size_column])
        
        try:
            # Perform regression of factor on log market cap
            X = df[['log_' + size_column]]
            X = sm.add_constant(X)  # Add constant
            
            # Use processed y values
            y = df[f'{factor_name}_processed']
            
            model = OLS(y, X).fit()
            
            # Calculate residuals (size-neutralized factor)
            df[f'{factor_name}_size_neutral'] = model.resid
            
            # Standardize the neutralized factor
            df[f'{factor_name}_size_neutral'] = self._standardize(df[f'{factor_name}_size_neutral'])
            
            # Return original DataFrame with neutralized factor added
            result = factor_df.copy()
            result = result.merge(
                df[[f'{factor_name}_size_neutral']],
                left_index=True,
                right_index=True,
                how='left'
            )
            
            return result
            
        except Exception as e:
            warnings.warn(f"Error in size neutralization: {str(e)}")
            return factor_df
    
    def neutralize_combined(self, factor_df, factor_name, size_column='dlycap', 
                       naics_column='naics_sector', winsorize_factor=True, 
                       standardize_factor=True, mad_n=3):
        """
        Neutralize a target factor with respect to size and NAICS industry.
        
        This performs a single regression against all neutralizing characteristics.
        
        Parameters:
        -----------
        factor_df : DataFrame
            DataFrame containing all factors and characteristics
        factor_name : str
            Name of the factor to be neutralized
        size_column : str, default 'dlycap'
            Name of the market cap column
        naics_column : str, default 'naics_sector'
            Name of the NAICS column
        winsorize_factor : bool, default True
            Whether to winsorize the factor before neutralization
        standardize_factor : bool, default True
            Whether to standardize the factor before neutralization
        mad_n : int, default 3
            Number of MADs to use for winsorization
            
        Returns:
        --------
        DataFrame
            DataFrame with the fully neutralized target added as a new column
        """
        # Create copy to avoid modifying the original
        df = factor_df.copy()
        
        # Apply winsorization and standardization if requested
        if winsorize_factor:
            y_raw = df[factor_name]
            y = self._winsorize(y_raw, n=mad_n)
        else:
            y = df[factor_name]
        
        if standardize_factor:
            y = self._standardize(y)
        
        # Store processed y values back in the dataframe
        df[f'{factor_name}_processed'] = y
        
        # Create NAICS dummy variables if NAICS column exists
        if naics_column in df.columns:
            df = self._create_dummy_variables(df, naics_column)
            naics_dummies = [col for col in df.columns if col.startswith(f"{naics_column}_")]
        else:
            naics_dummies = []
            
        # Handle market cap if size column exists
        if size_column in df.columns:
            # Handle missing or zero market cap values by replacing with mean
            valid_sizes = df[df[size_column] > 0][size_column]
            if len(valid_sizes) > 0:
                mean_size = valid_sizes.mean()
                df[size_column] = df[size_column].replace(0, np.nan).fillna(mean_size)
            
            # Log transform market cap
            df['log_' + size_column] = np.log(df[size_column])
            size_vars = ['log_' + size_column]
        else:
            size_vars = []
            
        # Combine all neutralizing variables
        all_neutralizing_vars = naics_dummies + size_vars
        
        if not all_neutralizing_vars:
            warnings.warn("No valid neutralizing variables. Returning original factor.")
            return factor_df
            
        # Handle missing values
        df = df.dropna(subset=[f'{factor_name}_processed'] + all_neutralizing_vars)
        
        try:
            # Perform regression of target factor on all neutralizing variables
            X = df[all_neutralizing_vars]
            X = sm.add_constant(X)  # Add constant
            
            # Use processed y values
            y = df[f'{factor_name}_processed']

            X = X.select_dtypes(include=['bool', 'object']).astype(int)
            
            model = OLS(y, X).fit()
            
            # Calculate residuals (fully neutralized target)
            df[f'{factor_name}_neutral'] = model.resid
            
            # Standardize the neutralized factor
            df[f'{factor_name}_neutral'] = self._standardize(df[f'{factor_name}_neutral'])
            
            # Return original DataFrame with neutralized factor added
            result = factor_df.copy()
            result = result.merge(
                df[[f'{factor_name}_neutral']],
                left_index=True,
                right_index=True,
                how='left'
            )
            
            return result
            
        except Exception as e:
            warnings.warn(f"Error in combined neutralization: {str(e)}")
            return factor_df
        


class FactorCalculator:
    """
    Base class for factor calculations. Provides common infrastructure and defines
    the interface that specific factor calculators must implement.
    """
    def __init__(self, factor_name):
        self.factor_name = factor_name
    
    def calculate(self, date_str, universe_df, duck_conn):
        """
        Abstract method to be implemented by specific factors.
        Should return a DataFrame containing factor values for the universe.
        """
        raise NotImplementedError
        
    def get_output_folder(self, base_folder):
        """Creates and returns the factor-specific output folder"""
        folder_path = f"{base_folder}/{self.factor_name}"
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

class ROECalculator(FactorCalculator):
    """
    Calculator for Return on Equity (ROE) factor.
    ROE = Income Before Extraordinary Items / Average Common Equity
    """
    def __init__(self):
        super().__init__("roe")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        ranked_data AS (
            SELECT *
            FROM fundamentals
            WHERE quarter_rank <= 2
        ),
        equity_calc AS (
            SELECT 
                f.gvkey,
                CASE 
                    WHEN COUNT(*) = 2 THEN AVG(CAST(f.ceqq AS DOUBLE))
                    ELSE MAX(CAST(f.ceqq AS DOUBLE))
                END as avg_ceqq,
                COUNT(*) as quarters_available
            FROM ranked_data f
            GROUP BY f.gvkey
            HAVING COUNT(*) >= 1
        ),
        current_income AS (
            SELECT 
                f.gvkey,
                CAST(f.ibcomq AS DOUBLE) as current_ibq
            FROM ranked_data f
            WHERE f.quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            CASE
                WHEN ec.avg_ceqq < 0 THEN NULL
                ELSE ci.current_ibq / NULLIF(ec.avg_ceqq, 0)
            END as roe,
            ci.current_ibq as income,
            ec.avg_ceqq as equity,
            ec.quarters_available
        FROM temp_universe u
        LEFT JOIN current_income ci
            ON ci.gvkey = u.gvkey
        LEFT JOIN equity_calc ec 
            ON ec.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(query.format(date=date_str)).fetchdf()
            
            # Calculate and display summary statistics
            self._print_summary_statistics(result, date_str)
            
            # Remove calculation columns before returning
            final_result = result.drop(['quarters_available', 'income', 'equity'], axis=1)
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Prints detailed summary statistics for the ROE calculation"""
        total_stocks = len(df)
        stocks_with_income = df['income'].notna().sum()
        stocks_with_equity = df['equity'].notna().sum()
        stocks_with_positive_equity = (df['equity'] > 0).sum()
        stocks_with_negative_equity = (df['equity'] < 0).sum()
        stocks_with_two_quarters = len(df[df['quarters_available'] == 2])
        stocks_with_one_quarter = len(df[df['quarters_available'] == 1])
        stocks_with_roe = df['roe'].notna().sum()
        
        print(f"\nROE Calculation Summary for {date_str}:")
        print(f"Total stocks in universe: {total_stocks}")
        print(f"Stocks with income data: {stocks_with_income}")
        print(f"Stocks with equity data: {stocks_with_equity}")
        print(f"- Using two quarters: {stocks_with_two_quarters}")
        print(f"- Using one quarter: {stocks_with_one_quarter}")
        print(f"- With positive equity: {stocks_with_positive_equity}")
        print(f"- With negative equity: {stocks_with_negative_equity}")
        print(f"Final stocks with valid ROE: {stocks_with_roe}")


class RelativeStrengthCalculator(FactorCalculator):
    def __init__(self, lookback_days):
        # Initialize with factor name for consistency
        factor_name = f"relative_strength_{lookback_days}d"
        super().__init__(factor_name)
        self.lookback_days = lookback_days
        self.factor_name = factor_name
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH date_ranges AS (
            SELECT 
                DATE '{date}' as end_date,
                DATE '{date}' - INTERVAL '{lookback} days' as start_date
        ),
        daily_returns AS (
            SELECT 
                s.permno,
                s.dlycaldt,
                s.DlyRet
            FROM stkdlysecuritydata s
            INNER JOIN temp_universe u ON u.permno = s.permno
            INNER JOIN date_ranges d 
                ON s.dlycaldt <= d.end_date 
                AND s.dlycaldt > d.start_date
            WHERE s.DlyRet IS NOT NULL
        ),
        cumulative_returns AS (
            SELECT 
                permno,
                COUNT(*) as num_observations,
                EXP(SUM(LN(1 + CAST(DlyRet AS DOUBLE)))) - 1 as cum_return
            FROM daily_returns
            GROUP BY permno
        ),
        ranked_returns AS (
            SELECT 
                permno,
                cum_return,
                num_observations,
                ROW_NUMBER() OVER (ORDER BY cum_return) as rank,
                COUNT(*) OVER () as total_stocks
            FROM cumulative_returns
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            r.cum_return,
            -- Name the column using the factor name for consistency
            (CAST(r.rank AS DOUBLE) - 1) / (r.total_stocks - 1) as {factor_name}
        FROM temp_universe u
        LEFT JOIN ranked_returns r ON r.permno = u.permno
        """
        
        try:
            result = duck_conn.execute(
                query.format(
                    date=date_str,
                    lookback=self.lookback_days,
                    factor_name=self.factor_name  # Use the factor name for the column
                )
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            # Remove the cum_return column but keep our properly named factor column
            final_result = result.drop(['cum_return'], axis=1)
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        total_stocks = len(df)
        # Use the factor name when accessing the column
        stocks_with_factor = df[self.factor_name].notna().sum()
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Total stocks in universe: {total_stocks}")
        print(f"Stocks with valid factor: {stocks_with_factor}")
        print(f"Coverage ratio: {(stocks_with_factor/total_stocks)*100:.2f}%")
        
        if stocks_with_factor > 0:
            print("\nFactor Distribution:")
            print(df[self.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))


class ROICCalculator(FactorCalculator):
    """
    Calculator for Return on Invested Capital (ROIC).
    
    ROIC = NOPAT / Invested Capital
    where:
    - NOPAT (Net Operating Profit After Taxes) represents the operating profit adjusted for taxes
    - Invested Capital represents the total investment in the business
    
    We need to make several adjustments to get an accurate measure:
    1. For NOPAT:
       - Start with Operating Income (OIADPQ)
       - Adjust for taxes using reported tax rate
    2. For Invested Capital:
       - Net Working Capital (current assets - current liabilities)
       - Plus Net Fixed Assets (PPENTQ)
       - Plus Other Operating Assets
    """
    def __init__(self):
        super().__init__("roic")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT *
            FROM fundamentals
            WHERE quarter_rank = 1
        ),
        nopat_calc AS (
            SELECT 
                gvkey,
                oiadpq,
                txtq,
                piq,
                CASE 
                    WHEN txtq > oiadpq THEN oiadpq * 0.65
                    ELSE oiadpq * (1 - CAST(txtq AS DOUBLE) / NULLIF(piq, 0))
                END as nopat
            FROM latest_quarter
        ),
        capital_calc AS (
            SELECT 
                gvkey,
                actq,
                lctq,
                COALESCE(ppentq, 0) as ppentq,
                (CAST(actq AS DOUBLE) - CAST(lctq AS DOUBLE) + CAST(ppentq AS DOUBLE)) as invested_capital
            FROM latest_quarter
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            -- Include intermediate calculation values
            n.oiadpq as operating_income,
            n.txtq as tax_expense,
            n.piq as pretax_income,
            n.nopat as nopat,
            c.actq as current_assets,
            c.lctq as current_liabilities,
            c.ppentq as fixed_assets,
            c.invested_capital as invested_capital,
            -- Final ROIC calculation
            CASE 
                WHEN nopat < 0 AND invested_capital < 0 THEN NULL
                ELSE CAST(n.nopat / NULLIF(c.invested_capital, 0) AS DOUBLE)
            END as roic
        FROM temp_universe u
        LEFT JOIN nopat_calc n ON n.gvkey = u.gvkey
        LEFT JOIN capital_calc c ON c.gvkey = u.gvkey
        """
        
        try:
            # Get result with all intermediate calculations
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            # Pass the full result to summary statistics for analysis
            self._print_summary_statistics(result, date_str)
            
            # Define columns to keep in final result
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'roic']
            
            # Return only the final columns
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')

    def _print_summary_statistics(self, df, date_str):
        """
        Print comprehensive analysis using intermediate calculation values.
        """
        print(f"\nROIC Calculation Summary for {date_str}")
        print("=" * 50)

        total_stocks = len(df)
        stocks_with_roic = df['roic'].notna().sum()
        
        print(f"\nOverall Coverage:")
        print(f"Total stocks in universe: {total_stocks}")
        print(f"Stocks with valid ROIC: {stocks_with_roic}")
        print(f"Coverage ratio: {(stocks_with_roic/total_stocks)*100:.2f}%")

        print("\nComponent Availability Analysis:")
        components = {
            'Operating Income': 'operating_income',
            'Tax Expense': 'tax_expense',
            'Pretax Income': 'pretax_income',
            'NOPAT': 'nopat',
            'Current Assets': 'current_assets',
            'Current Liabilities': 'current_liabilities',
            'Fixed Assets': 'fixed_assets',
            'Invested Capital': 'invested_capital'
        }

        for label, column in components.items():
            available = df[column].notna().sum()
            print(f"{label}: {available} stocks ({available/total_stocks*100:.1f}%)")

        print("\nComponent Statistics:")
        if df['nopat'].notna().any():
            print("\nNOPAT Distribution:")
            print(df['nopat'].describe(percentiles=[.05, .25, .5, .75, .95]))

        if df['invested_capital'].notna().any():
            print("\nInvested Capital Distribution:")
            print(df['invested_capital'].describe(percentiles=[.05, .25, .5, .75, .95]))

        print("\nROIC Distribution:")
        if stocks_with_roic > 0:
            print(df['roic'].describe(percentiles=[.05, .25, .5, .75, .95]))


class PriceToSalesCalculator(FactorCalculator):
    """
    Calculator for Price-to-Sales ratio.
    P/S = Market Cap / Sales
    Using:
    - Market Cap: DlyCap from CRSP daily stock data
    - Sales: Most recent quarter's sales (saleq)
    """
    def __init__(self):
        super().__init__("price_to_sales")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            -- Get most recent quarter's sales
            SELECT 
                gvkey,
                CAST(saleq AS DOUBLE) as sales_qtr
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            s.DlyCap as market_cap,
            q.sales_qtr,
            -- Calculate P/S ratio 
            s.DlyCap / NULLIF(q.sales_qtr, 0)
            as price_to_sales
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            # Keep intermediate calculations for analysis
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'price_to_sales']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of P/S calculation results"""
        total_stocks = len(df)
        stocks_with_ps = df['price_to_sales'].notna().sum()
        
        print(f"\nPrice-to-Sales Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid P/S: {stocks_with_ps}")
        print(f"Coverage ratio: {(stocks_with_ps/total_stocks)*100:.2f}%")
        
        if stocks_with_ps > 0:
            print("\nQuarterly Sales Distribution (millions):")
            print(df['sales_qtr'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nP/S Ratio Distribution:")
            print(df['price_to_sales'].describe(percentiles=[.05, .25, .5, .75, .95]))


class CashROICCalculator(FactorCalculator):
    """
    Calculator for Cash ROIC using quarterly operating cash flow.
    
    Cash ROIC = Quarterly Operating Cash Flow / Invested Capital
    where:
    - Operating Cash Flow: Calculate quarterly from OANCFY (YTD Operating Cash Flow)
        - For Q1: Use YTD directly 
        - For Q2-Q4: Current YTD minus previous quarter YTD if available
    - Invested Capital = Total Assets - Current Liabilities - Cash
    """
    def __init__(self):
        super().__init__("cash_roic")
   
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH ranked_fundamentals AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY gvkey 
                    ORDER BY datadate DESC
                ) as quarter_rank,
                LAG(oancfy) OVER (
                    PARTITION BY gvkey 
                    ORDER BY datadate
                ) as prev_oancfy
            FROM wrds_csq_pit
            WHERE 
                pitdate1 <= DATE '{date}'
                AND (pitdate2 >= DATE '{date}' OR pitdate2 IS NULL)
                AND indfmt = 'INDL'
                AND datafmt = 'STD'
                AND consol = 'C'
                AND popsrc = 'D'
                AND curcdq = 'USD'
                AND updq = 3
        ),
        latest_data AS (
            SELECT *
            FROM ranked_fundamentals
            WHERE quarter_rank = 1
        ),
        cash_flow_calc AS (
            SELECT 
                gvkey,
                CASE 
                    WHEN fqtr = 1 THEN CAST(oancfy AS DOUBLE)
                    WHEN prev_oancfy IS NULL THEN CAST(oancfy AS DOUBLE)
                    ELSE CAST(oancfy AS DOUBLE) - CAST(prev_oancfy AS DOUBLE)
                END as quarterly_cash_flow
            FROM latest_data
        ),
        capital_calc AS (
            SELECT 
                gvkey,
                (CAST(atq AS DOUBLE) - CAST(lctq AS DOUBLE) - COALESCE(CAST(cheq AS DOUBLE), 0)) as invested_capital
            FROM latest_data
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            f.quarterly_cash_flow,
            c.invested_capital,
            CASE 
                WHEN f.quarterly_cash_flow < 0 AND c.invested_capital < 0 THEN NULL
                ELSE CAST(f.quarterly_cash_flow / NULLIF(c.invested_capital, 0) AS DOUBLE)
            END as cash_roic
        FROM temp_universe u
        LEFT JOIN cash_flow_calc f ON f.gvkey = u.gvkey
        LEFT JOIN capital_calc c ON c.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'cash_roic']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
   
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Cash ROIC calculation results"""
        total_stocks = len(df)
        stocks_with_roic = df['cash_roic'].notna().sum()

        # Calculate missing percentages for quarterly_cash_flow and invested_capital
        missing_quarterly_cash_flow = df['quarterly_cash_flow'].isna().mean() * 100
        missing_invested_capital = df['invested_capital'].isna().mean() * 100

        print(f"\nCash ROIC Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Cash ROIC: {stocks_with_roic}")
        print(f"Coverage ratio: {(stocks_with_roic / total_stocks) * 100:.2f}%")
        print(f"Missing Quarterly Cash Flow: {missing_quarterly_cash_flow:.2f}%")
        print(f"Missing Invested Capital: {missing_invested_capital:.2f}%")

        if stocks_with_roic > 0:
            print("\nQuarterly Cash Flow Distribution:")
            print(df['quarterly_cash_flow'].describe(percentiles=[.05, .25, .5, .75, .95]))

            print("\nInvested Capital Distribution:")
            print(df['invested_capital'].describe(percentiles=[.05, .25, .5, .75, .95]))

            print("\nCash ROIC Distribution:")
            print(df['cash_roic'].describe(percentiles=[.05, .25, .5, .75, .95]))

class PriceToBookCalculator(FactorCalculator):
    """
    Calculator for Price-to-Book ratio (P/B).
    P/B = Market Cap / Book Value of Equity
    
    Compustat fields:
    - Market Cap: Using dlycap from CRSP
    - Book Value: Common/Ordinary Equity (ceqq)
    """
    def __init__(self):
        super().__init__("price_to_book")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(ceqq AS DOUBLE) as book_value
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            s.dlycap as market_cap,
            q.book_value,
            -- Calculate P/B ratio 
            s.dlycap / NULLIF(q.book_value, 0)
            as price_to_book
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'price_to_book']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of P/B calculation results"""
        total_stocks = len(df)
        stocks_with_pb = df['price_to_book'].notna().sum()
        
        print(f"\nPrice-to-Book Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid P/B: {stocks_with_pb}")
        print(f"Coverage ratio: {(stocks_with_pb/total_stocks)*100:.2f}%")
        
        if stocks_with_pb > 0:
            print("\nBook Value Distribution (millions):")
            print(df['book_value'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nP/B Ratio Distribution:")
            print(df['price_to_book'].describe(percentiles=[.05, .25, .5, .75, .95]))


class AdjustedProfitToProfitCalculator(FactorCalculator):
    """
    Calculator for Adjusted Profit to Profit ratio.
    adjusted_profit_to_profit = adjusted_profit / net_profit (%)
    
    This ratio measures the percentage of net income that comes from core operations
    (excluding extraordinary items and non-recurring gains/losses).
    
    Compustat fields:
    - Adjusted profit: ibcomq (Income Before Extraordinary Items - Available for Common)
    - Net profit: niq (Net Income)
    """
    def __init__(self):
        super().__init__("adjusted_profit_to_profit")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(ibcomq AS DOUBLE) as adjusted_profit,
                CAST(niq AS DOUBLE) as net_profit,
                
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.adjusted_profit,
            q.net_profit,
            -- Calculate quarterly adjusted_profit_to_profit
            CASE 
                WHEN q.net_profit = 0 THEN NULL
                ELSE q.adjusted_profit / q.net_profit
            END as adjusted_profit_to_profit,
            
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'adjusted_profit_to_profit']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Adjusted Profit to Profit ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['adjusted_profit_to_profit'].notna().sum()
        
        print(f"\nAdjusted Profit to Profit Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Ratio: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nAdjusted Profit Distribution (millions):")
            print(df['adjusted_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nNet Profit Distribution (millions):")
            print(df['net_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nAdjusted Profit to Profit Ratio Distribution:")
            print(df['adjusted_profit_to_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))


class OCFToRevenueCalculator(FactorCalculator):
    """
    Calculator for Operating Cash Flow to Revenue ratio.
    ocf_to_revenue = Operating Cash Flow / Revenue
    
    This ratio measures the ability of a company to convert its revenue into cash.
    
    Compustat fields:
    - Operating Cash Flow: oancfy (operating activities - net cash flow)
    - Revenue: saleq (quarterly sales)
    """
    def __init__(self):
        super().__init__("ocf_to_revenue")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank,
                LAG(oancfy) OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate
                ) as prev_oancfy
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_data AS (
            SELECT 
                gvkey,
                CASE 
                    WHEN fqtr = 1 THEN CAST(oancfy AS DOUBLE)
                    WHEN prev_oancfy IS NULL THEN CAST(oancfy AS DOUBLE)
                    ELSE CAST(oancfy AS DOUBLE) - CAST(prev_oancfy AS DOUBLE)
                END as quarterly_ocf,
                CAST(saleq AS DOUBLE) as revenue
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.quarterly_ocf,
            q.revenue,
            -- Calculate Operating Cash Flow to Revenue ratio
            CASE 
                WHEN q.revenue = 0 THEN NULL
                ELSE q.quarterly_ocf / q.revenue
            END as ocf_to_revenue
        FROM temp_universe u
        LEFT JOIN latest_data q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'ocf_to_revenue']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of OCF to Revenue ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['ocf_to_revenue'].notna().sum()
        
        print(f"\nOperating Cash Flow to Revenue Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Ratio: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nOperating Cash Flow Distribution (millions):")
            print(df['quarterly_ocf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nOCF to Revenue Ratio Distribution:")
            print(df['ocf_to_revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))

        
class OperatingProfitToProfitCalculator(FactorCalculator):
    """
    Calculator for Operating Profit to Profit ratio.
    operating_profit_to_profit = Operating Profit / Net Profit
    
    This ratio measures the proportion of a company's net income that is derived from its core operating activities.
    
    Compustat fields:
    - Operating Profit: oiadpq (Operating Income After Depreciation)
    - Net Profit: niq (Net Income)
    """
    def __init__(self):
        super().__init__("operating_profit_to_profit")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(oiadpq AS DOUBLE) as operating_profit,
                CAST(niq AS DOUBLE) as net_profit
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.operating_profit,
            q.net_profit,
            -- Calculate Operating Profit to Profit ratio
            CASE 
                WHEN q.net_profit = 0 THEN NULL
                ELSE q.operating_profit / q.net_profit
            END as operating_profit_to_profit
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'operating_profit_to_profit']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Operating Profit to Profit ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['operating_profit_to_profit'].notna().sum()
        
        print(f"\nOperating Profit to Profit Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Ratio: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nOperating Profit Distribution (millions):")
            print(df['operating_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nNet Profit Distribution (millions):")
            print(df['net_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nOperating Profit to Profit Ratio Distribution:")
            print(df['operating_profit_to_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))


class CurrentDebtToTotalDebtCalculator(FactorCalculator):
    """
    Calculator for Current Debt to Total Debt ratio.
    current_debt_to_total_debt = Current Liabilities / Total Liabilities
    
    This ratio measures what portion of a company's debt is due in the short term.
    
    Compustat fields:
    - Current Liabilities: lctq
    - Total Liabilities: ltq
    """
    def __init__(self):
        super().__init__("current_debt_to_total_debt")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(lctq AS DOUBLE) as current_liabilities,
                CAST(ltq AS DOUBLE) as total_liabilities
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.current_liabilities,
            q.total_liabilities,
            -- Calculate Current Debt to Total Debt ratio
            CASE 
                WHEN q.total_liabilities = 0 THEN NULL
                ELSE q.current_liabilities / q.total_liabilities
            END as current_debt_to_total_debt
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'current_debt_to_total_debt']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Current Debt to Total Debt ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['current_debt_to_total_debt'].notna().sum()
        
        print(f"\nCurrent Debt to Total Debt Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Ratio: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nCurrent Liabilities Distribution (millions):")
            print(df['current_liabilities'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTotal Liabilities Distribution (millions):")
            print(df['total_liabilities'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCurrent Debt to Total Debt Ratio Distribution:")
            print(df['current_debt_to_total_debt'].describe(percentiles=[.05, .25, .5, .75, .95]))


class CurrentAssetsToCurrentDebtCalculator(FactorCalculator):
    """
    Calculator for Current Assets to Current Debt ratio (Current Ratio).
    current_assets_to_current_debt = Current Assets / Current Liabilities
    
    This ratio measures a company's ability to pay short-term obligations.
    
    Compustat fields:
    - Current Assets: actq
    - Current Liabilities: lctq
    """
    def __init__(self):
        super().__init__("current_assets_to_current_debt")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(actq AS DOUBLE) as current_assets,
                CAST(lctq AS DOUBLE) as current_liabilities
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.current_assets,
            q.current_liabilities,
            -- Calculate Current Assets to Current Debt ratio
            q.current_assets / NULLIF(q.current_liabilities, 0) as current_assets_to_current_debt
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'current_assets_to_current_debt']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Current Assets to Current Debt ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['current_assets_to_current_debt'].notna().sum()
        
        print(f"\nCurrent Assets to Current Debt Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Ratio: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nCurrent Assets Distribution (millions):")
            print(df['current_assets'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCurrent Liabilities Distribution (millions):")
            print(df['current_liabilities'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCurrent Assets to Current Debt Ratio Distribution:")
            print(df['current_assets_to_current_debt'].describe(percentiles=[.05, .25, .5, .75, .95]))


class NetProfitMarginCalculator(FactorCalculator):
    """
    Calculator for Net Profit Margin.
    Net Profit Margin = Net Income / Revenue
    
    Compustat fields:
    - Net Income: niq (quarterly net income)
    - Revenue: saleq (quarterly sales)
    """
    def __init__(self):
        super().__init__("net_profit_margin")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(niq AS DOUBLE) as net_income,
                CAST(saleq AS DOUBLE) as revenue
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.net_income,
            q.revenue,
            -- Calculate Net Profit Margin
            CASE 
                WHEN q.revenue <= 0 THEN NULL
                ELSE q.net_income / q.revenue
            END as net_profit_margin
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'net_profit_margin']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Net Profit Margin calculation"""
        total_stocks = len(df)
        stocks_with_npm = df['net_profit_margin'].notna().sum()
        
        print(f"\nNet Profit Margin Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Net Profit Margin: {stocks_with_npm}")
        print(f"Coverage ratio: {(stocks_with_npm/total_stocks)*100:.2f}%")
        
        if stocks_with_npm > 0:
            print("\nNet Income Distribution (millions):")
            print(df['net_income'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nNet Profit Margin Distribution:")
            print(df['net_profit_margin'].describe(percentiles=[.05, .25, .5, .75, .95]))

class ROACalculator(FactorCalculator):
    """
    Calculator for Return on Assets (ROA).
    ROA = (Quarterly Net Income * 4) / Average Total Assets
    
    This ratio measures how efficiently a company uses its assets to generate profits.
    
    Compustat fields:
    - Net Income: niq (quarterly net income)
    - Total Assets: atq
    """
    def __init__(self):
        super().__init__("roa")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        ranked_data AS (
            SELECT *
            FROM fundamentals
            WHERE quarter_rank <= 2
        ),
        asset_calc AS (
            SELECT 
                gvkey,
                CASE 
                    WHEN COUNT(*) = 2 THEN AVG(CAST(atq AS DOUBLE))
                    ELSE MAX(CAST(atq AS DOUBLE))
                END as avg_assets,
                COUNT(*) as quarters_available
            FROM ranked_data
            GROUP BY gvkey
            HAVING COUNT(*) >= 1
        ),
        income_data AS (
            SELECT 
                gvkey,
                CAST(niq AS DOUBLE) as quarterly_income
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            i.quarterly_income,
            a.avg_assets,
            a.quarters_available,
            -- Calculate quarterly ROA (annualized)
            (i.quarterly_income * 4) / NULLIF(a.avg_assets, 0) as roa
        FROM temp_universe u
        LEFT JOIN income_data i ON i.gvkey = u.gvkey
        LEFT JOIN asset_calc a ON a.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'roa']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of ROA calculation"""
        total_stocks = len(df)
        stocks_with_roa = df['roa'].notna().sum()
        
        print(f"\nReturn on Assets (ROA) Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with calculated ROA: {stocks_with_roa}")
        print(f"Coverage ratio: {(stocks_with_roa/total_stocks)*100:.2f}%")
        print(f"Stocks with two quarters of asset data: {len(df[df['quarters_available'] == 2])}")
        print(f"Stocks with one quarter of asset data: {len(df[df['quarters_available'] == 1])}")
        
        if stocks_with_roa > 0:
            print("\nQuarterly Income Distribution (millions):")
            print(df['quarterly_income'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nAverage Assets Distribution (millions):")
            print(df['avg_assets'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nROA Distribution:")
            print(df['roa'].describe(percentiles=[.05, .25, .5, .75, .95]))



class AssetTurnoverRateCalculator(FactorCalculator):
    """
    Calculator for Total Asset Turnover Rate.
    total_asset_turnover_rate = (Quarterly Revenue * 4) / Average Total Assets
    
    This ratio measures how efficiently a company uses its assets to generate sales.
    
    Compustat fields:
    - Revenue: saleq (quarterly sales)
    - Total Assets: atq
    """
    def __init__(self):
        super().__init__("total_asset_turnover_rate")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        ranked_data AS (
            SELECT *
            FROM fundamentals
            WHERE quarter_rank <= 2
        ),
        asset_calc AS (
            SELECT 
                gvkey,
                CASE 
                    WHEN COUNT(*) = 2 THEN AVG(CAST(atq AS DOUBLE))
                    ELSE MAX(CAST(atq AS DOUBLE))
                END as avg_assets,
                COUNT(*) as quarters_available
            FROM ranked_data
            GROUP BY gvkey
            HAVING COUNT(*) >= 1
        ),
        current_revenue AS (
            SELECT 
                gvkey,
                CAST(saleq AS DOUBLE) as quarterly_revenue
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            r.quarterly_revenue,
            a.avg_assets,
            a.quarters_available,
            -- Calculate Asset Turnover (annualized by multiplying quarterly revenue by 4)
            (r.quarterly_revenue * 4) / NULLIF(a.avg_assets, 0) as total_asset_turnover_rate
        FROM temp_universe u
        LEFT JOIN current_revenue r ON r.gvkey = u.gvkey
        LEFT JOIN asset_calc a ON a.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'total_asset_turnover_rate']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Total Asset Turnover Rate calculation"""
        total_stocks = len(df)
        stocks_with_turnover = df['total_asset_turnover_rate'].notna().sum()
        
        print(f"\nTotal Asset Turnover Rate Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Turnover Rate: {stocks_with_turnover}")
        print(f"Coverage ratio: {(stocks_with_turnover/total_stocks)*100:.2f}%")
        print(f"Stocks with two quarters of asset data: {len(df[df['quarters_available'] == 2])}")
        print(f"Stocks with one quarter of asset data: {len(df[df['quarters_available'] == 1])}")
        
        if stocks_with_turnover > 0:
            print("\nQuarterly Revenue Distribution (millions):")
            print(df['quarterly_revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nAverage Assets Distribution (millions):")
            print(df['avg_assets'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTotal Asset Turnover Rate Distribution:")
            print(df['total_asset_turnover_rate'].describe(percentiles=[.05, .25, .5, .75, .95]))
    
class OperatingProfitGrowthCalculator(FactorCalculator):
    """
    Calculator for Year-on-Year Operating Profit Growth.
    inc_operation_profit_year_on_year = (Operating Profit Current - Operating Profit Last Year) / |Operating Profit Last Year|
    
    This measures the percentage change in operating profit compared to the same quarter last year.
    
    Compustat fields:
    - Operating Profit: oiadpq (Operating Income After Depreciation)
    """
    def __init__(self):
        super().__init__("inc_operation_profit_year_on_year")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        quarterly_data AS (
            SELECT 
                gvkey,
                datadate,
                fqtr,
                fyearq,
                quarter_rank,
                CAST(oiadpq AS DOUBLE) as operating_profit
            FROM fundamentals
            WHERE quarter_rank <= 5  -- Get 5 quarters of data
        ),
        compare_quarters AS (
            SELECT 
                a.gvkey,
                a.datadate as current_date,
                a.fqtr as quarter,
                a.operating_profit as current_profit,
                b.operating_profit as previous_year_profit,
                -- Calculate Year-on-Year Operating Profit Growth
                CASE 
                    WHEN b.operating_profit = 0 THEN NULL
                    ELSE (a.operating_profit - b.operating_profit) / ABS(b.operating_profit)
                END as inc_operation_profit_year_on_year
            FROM 
                (SELECT * FROM quarterly_data WHERE quarter_rank = 1) a  -- Most recent quarter (Q1)
            LEFT JOIN 
                quarterly_data b 
                -- Match to the same quarter one year ago (Q5)
                ON a.gvkey = b.gvkey 
                AND a.fqtr = b.fqtr 
                AND a.fyearq = b.fyearq + 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            c.current_profit,
            c.previous_year_profit,
            c.quarter,
            c.inc_operation_profit_year_on_year
        FROM temp_universe u
        LEFT JOIN compare_quarters c ON c.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'inc_operation_profit_year_on_year']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Year-on-Year Operating Profit Growth calculation"""
        total_stocks = len(df)
        stocks_with_growth = df['inc_operation_profit_year_on_year'].notna().sum()
        
        print(f"\nYear-on-Year Operating Profit Growth Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Growth Rate: {stocks_with_growth}")
        print(f"Coverage ratio: {(stocks_with_growth/total_stocks)*100:.2f}%")
        
        if stocks_with_growth > 0:
            print("\nCurrent Quarter Operating Profit Distribution (millions):")
            print(df['current_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nPrevious Year Same Quarter Operating Profit Distribution (millions):")
            print(df['previous_year_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nYear-on-Year Operating Profit Growth Distribution:")
            print(df['inc_operation_profit_year_on_year'].describe(percentiles=[.05, .25, .5, .75, .95]))


class NetProfitToShareholdersGrowthCalculator(FactorCalculator):
    """
    Calculator for Year-on-Year Net Profit to Shareholders Growth.
    inc_net_profit_to_shareholders_year_on_year = (Net Profit to Shareholders Current - Net Profit to Shareholders Last Year) / |Net Profit to Shareholders Last Year|
    
    This measures the percentage change in net profit attributable to shareholders compared to the same quarter last year.
    
    Compustat fields:
    - Net Profit to Shareholders: ibcomq (Income Before Extraordinary Items - Available for Common)
    """
    def __init__(self):
        super().__init__("inc_net_profit_to_shareholders_year_on_year")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        quarterly_data AS (
            SELECT 
                gvkey,
                datadate,
                fqtr,
                fyearq,
                quarter_rank,
                CAST(ibcomq AS DOUBLE) as net_profit_to_shareholders
            FROM fundamentals
            WHERE quarter_rank <= 5  -- Get 5 quarters of data
        ),
        compare_quarters AS (
            SELECT 
                a.gvkey,
                a.datadate as current_date,
                a.fqtr as quarter,
                a.net_profit_to_shareholders as current_profit,
                b.net_profit_to_shareholders as previous_year_profit,
                -- Calculate Year-on-Year Net Profit to Shareholders Growth
                CASE 
                    WHEN b.net_profit_to_shareholders = 0 THEN NULL
                    ELSE (a.net_profit_to_shareholders - b.net_profit_to_shareholders) / ABS(b.net_profit_to_shareholders)
                END as inc_net_profit_to_shareholders_year_on_year
            FROM 
                (SELECT * FROM quarterly_data WHERE quarter_rank = 1) a  -- Most recent quarter (Q1)
            LEFT JOIN 
                quarterly_data b 
                -- Match to the same quarter one year ago (Q5)
                ON a.gvkey = b.gvkey 
                AND a.fqtr = b.fqtr 
                AND a.fyearq = b.fyearq + 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            c.current_profit,
            c.previous_year_profit,
            c.quarter,
            c.inc_net_profit_to_shareholders_year_on_year
        FROM temp_universe u
        LEFT JOIN compare_quarters c ON c.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'inc_net_profit_to_shareholders_year_on_year']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Year-on-Year Net Profit to Shareholders Growth calculation"""
        total_stocks = len(df)
        stocks_with_growth = df['inc_net_profit_to_shareholders_year_on_year'].notna().sum()
        
        print(f"\nYear-on-Year Net Profit to Shareholders Growth Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Growth Rate: {stocks_with_growth}")
        print(f"Coverage ratio: {(stocks_with_growth/total_stocks)*100:.2f}%")
        
        if stocks_with_growth > 0:
            print("\nCurrent Quarter Net Profit to Shareholders Distribution (millions):")
            print(df['current_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nPrevious Year Same Quarter Net Profit to Shareholders Distribution (millions):")
            print(df['previous_year_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nYear-on-Year Net Profit to Shareholders Growth Distribution:")
            print(df['inc_net_profit_to_shareholders_year_on_year'].describe(percentiles=[.05, .25, .5, .75, .95]))


class NetProfitGrowthCalculator(FactorCalculator):
    """
    Calculator for Year-on-Year Net Profit Growth.
    inc_net_profit_year_on_year = (Net Profit Current - Net Profit Last Year) / |Net Profit Last Year|
    
    This measures the percentage change in total net profit compared to the same quarter last year.
    
    Compustat fields:
    - Net Profit: niq (Net Income)
    """
    def __init__(self):
        super().__init__("inc_net_profit_year_on_year")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        quarterly_data AS (
            SELECT 
                gvkey,
                datadate,
                fqtr,
                fyearq,
                quarter_rank,
                CAST(niq AS DOUBLE) as net_profit
            FROM fundamentals
            WHERE quarter_rank <= 5  -- Get 5 quarters of data
        ),
        compare_quarters AS (
            SELECT 
                a.gvkey,
                a.datadate as current_date,
                a.fqtr as quarter,
                a.net_profit as current_profit,
                b.net_profit as previous_year_profit,
                -- Calculate Year-on-Year Net Profit Growth
                CASE 
                    WHEN b.net_profit = 0 THEN NULL
                    ELSE (a.net_profit - b.net_profit) / ABS(b.net_profit)
                END as inc_net_profit_year_on_year
            FROM 
                (SELECT * FROM quarterly_data WHERE quarter_rank = 1) a  -- Most recent quarter (Q1)
            LEFT JOIN 
                quarterly_data b 
                -- Match to the same quarter one year ago (Q5)
                ON a.gvkey = b.gvkey 
                AND a.fqtr = b.fqtr 
                AND a.fyearq = b.fyearq + 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            c.current_profit,
            c.previous_year_profit,
            c.quarter,
            c.inc_net_profit_year_on_year
        FROM temp_universe u
        LEFT JOIN compare_quarters c ON c.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'inc_net_profit_year_on_year']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Year-on-Year Net Profit Growth calculation"""
        total_stocks = len(df)
        stocks_with_growth = df['inc_net_profit_year_on_year'].notna().sum()
        
        print(f"\nYear-on-Year Net Profit Growth Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Growth Rate: {stocks_with_growth}")
        print(f"Coverage ratio: {(stocks_with_growth/total_stocks)*100:.2f}%")
        
        if stocks_with_growth > 0:
            print("\nCurrent Quarter Net Profit Distribution (millions):")
            print(df['current_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nPrevious Year Same Quarter Net Profit Distribution (millions):")
            print(df['previous_year_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nYear-on-Year Net Profit Growth Distribution:")
            print(df['inc_net_profit_year_on_year'].describe(percentiles=[.05, .25, .5, .75, .95]))


class EnterpriseValueCalculator(FactorCalculator):
    """
    Calculator for Enterprise Value (EV).
    EV = Market Cap + Net Debt
    
    Enterprise Value represents the total value of a company including debt.
    
    Compustat fields:
    - Market Cap: from CRSP (dlycap)
    - Net Debt: dlttq (long-term debt) + dlcq (current debt) - cheq (cash and equivalents)
    """
    def __init__(self):
        super().__init__("enterprise_value")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                COALESCE(CAST(dlttq AS DOUBLE), 0) as long_term_debt,
                COALESCE(CAST(dlcq AS DOUBLE), 0) as current_debt,
                COALESCE(CAST(cheq AS DOUBLE), 0) as cash_equivalents
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            s.dlycap as market_cap,
            q.long_term_debt,
            q.current_debt,
            q.cash_equivalents,
            -- Calculate Net Debt
            (q.long_term_debt + q.current_debt - q.cash_equivalents) as net_debt,
            -- Calculate Enterprise Value
            s.dlycap + (q.long_term_debt + q.current_debt - q.cash_equivalents) as enterprise_value
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'market_cap', 'net_debt', 'enterprise_value']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Enterprise Value calculation"""
        total_stocks = len(df)
        stocks_with_ev = df['enterprise_value'].notna().sum()
        
        print(f"\nEnterprise Value Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid EV: {stocks_with_ev}")
        print(f"Coverage ratio: {(stocks_with_ev/total_stocks)*100:.2f}%")
        
        if stocks_with_ev > 0:
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nNet Debt Distribution (millions):")
            print(df['net_debt'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nEnterprise Value Distribution (millions):")
            print(df['enterprise_value'].describe(percentiles=[.05, .25, .5, .75, .95]))


class SalesToEVCalculator(FactorCalculator):
    """
    Calculator for Sales to Enterprise Value ratio (S2EV).
    S2EV = Revenue / Enterprise Value
    
    This ratio measures how efficiently a company generates revenue relative to its total value.
    
    Compustat fields:
    - Revenue: saleq (quarterly sales)
    - Enterprise Value: calculated as market cap + net debt
    """
    def __init__(self):
        super().__init__("sales_to_ev")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(saleq AS DOUBLE) as revenue,
                COALESCE(CAST(dlttq AS DOUBLE), 0) as long_term_debt,
                COALESCE(CAST(dlcq AS DOUBLE), 0) as current_debt,
                COALESCE(CAST(cheq AS DOUBLE), 0) as cash_equivalents
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.revenue,
            s.dlycap as market_cap,
            (q.long_term_debt + q.current_debt - q.cash_equivalents) as net_debt,
            -- Calculate Enterprise Value
            (s.dlycap + (q.long_term_debt + q.current_debt - q.cash_equivalents)) as enterprise_value,
            -- Calculate Sales to Enterprise Value ratio
            q.revenue / NULLIF(s.dlycap + (q.long_term_debt + q.current_debt - q.cash_equivalents), 0) as sales_to_ev
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'sales_to_ev']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Sales to Enterprise Value ratio calculation"""
        total_stocks = len(df)
        stocks_with_s2ev = df['sales_to_ev'].notna().sum()
        
        print(f"\nSales to Enterprise Value Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid S2EV: {stocks_with_s2ev}")
        print(f"Coverage ratio: {(stocks_with_s2ev/total_stocks)*100:.2f}%")
        
        if stocks_with_s2ev > 0:
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nEnterprise Value Distribution (millions):")
            print(df['enterprise_value'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nSales to Enterprise Value Ratio Distribution:")
            print(df['sales_to_ev'].describe(percentiles=[.05, .25, .5, .75, .95]))


class BookToMarketCalculator(FactorCalculator):
    """
    Calculator for Book-to-Market ratio (BP) using the Fama-French (1993) approach.
    
    BP = Book Value / Market Cap
    
    Book Value = Shareholders' Equity + Deferred Taxes - Preferred Stock
    
    Compustat fields:
    - Shareholders' Equity: seqq (stockholders' equity)
    - Deferred Taxes: txditcq (deferred taxes and investment tax credit)
    - Preferred Stock: pstkq (preferred stock) or pstkrq (preferred stock redemption value)
    - Market Cap: from CRSP (dlycap)
    """
    def __init__(self):
        super().__init__("book_to_price")
        
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                -- Calculate book value following Fama-French (1993) methodology
                -- Use seqq (stockholders' equity)
                CAST(seqq AS DOUBLE) +
                -- Add deferred taxes (txditcq)
                COALESCE(CAST(txditcq AS DOUBLE), 0) -
                -- Subtract preferred stock (use redemption value if available, otherwise use par value)
                COALESCE(CAST(pstkrq AS DOUBLE), COALESCE(CAST(pstkq AS DOUBLE), 0)) AS book_value,
                datadate
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.book_value,
            s.dlycap as market_cap,
            -- Calculate Book-to-Price ratio (inverse of P/B)
            CASE 
                WHEN s.dlycap <= 0 THEN NULL
                WHEN q.book_value <= 0 THEN NULL  -- Exclude negative book values
                ELSE q.book_value / s.dlycap
            END as book_to_price,
            q.datadate as financial_statement_date
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'book_to_price']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Book-to-Price ratio calculation"""
        total_stocks = len(df)
        stocks_with_bp = df['book_to_price'].notna().sum()
        
        print(f"\nBook-to-Price Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid B/P: {stocks_with_bp}")
        print(f"Coverage ratio: {(stocks_with_bp/total_stocks)*100:.2f}%")
        
        if stocks_with_bp > 0:
            print("\nBook Value Distribution (millions):")
            print(df['book_value'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nBook-to-Price Ratio Distribution:")
            print(df['book_to_price'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            # Calculate median age of financial data
            if 'financial_statement_date' in df.columns:
                df['data_age_days'] = (pd.to_datetime(date_str) - pd.to_datetime(df['financial_statement_date'])).dt.days
                print("\nFinancial Statement Age (days):")
                print(df['data_age_days'].describe(percentiles=[.05, .25, .5, .75, .95]))


class CashFlowToAssetCalculator(FactorCalculator):
    """
    Calculator for Operating Cash Flow to Assets ratio.
    OCF to Assets = Operating Cash Flow / Total Assets
    
    This ratio measures a company's ability to generate cash from its assets.
    
    Compustat fields:
    - Operating Cash Flow: oancfy (operating activities - net cash flow)
    - Total Assets: atq
    """
    def __init__(self):
        super().__init__("cash_flow_to_asset")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank,
                LAG(oancfy) OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate
                ) as prev_oancfy
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_data AS (
            SELECT 
                gvkey,
                CASE 
                    WHEN fqtr = 1 THEN CAST(oancfy AS DOUBLE)
                    WHEN prev_oancfy IS NULL THEN CAST(oancfy AS DOUBLE)
                    ELSE CAST(oancfy AS DOUBLE) - CAST(prev_oancfy AS DOUBLE)
                END as quarterly_ocf,
                CAST(atq AS DOUBLE) as total_assets
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.quarterly_ocf,
            q.total_assets,
            -- Calculate Cash Flow to Asset ratio
            q.quarterly_ocf / NULLIF(q.total_assets, 0) as cash_flow_to_asset
        FROM temp_universe u
        LEFT JOIN latest_data q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'cash_flow_to_asset']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Cash Flow to Asset ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['cash_flow_to_asset'].notna().sum()
        
        print(f"\nCash Flow to Asset Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid CF/Asset: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nQuarterly Operating Cash Flow Distribution (millions):")
            print(df['quarterly_ocf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTotal Assets Distribution (millions):")
            print(df['total_assets'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCash Flow to Asset Ratio Distribution:")
            print(df['cash_flow_to_asset'].describe(percentiles=[.05, .25, .5, .75, .95]))


class CashFlowToLiabilityCalculator(FactorCalculator):
    """
    Calculator for Operating Cash Flow to Liability ratio.
    OCF to Liability = Operating Cash Flow / Total Liabilities
    
    This ratio measures a company's ability to pay its debts using operating cash flow.
    
    Compustat fields:
    - Operating Cash Flow: oancfy (operating activities - net cash flow)
    - Total Liabilities: ltq
    """
    def __init__(self):
        super().__init__("cash_flow_to_liability")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank,
                LAG(oancfy) OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate
                ) as prev_oancfy
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_data AS (
            SELECT 
                gvkey,
                CASE 
                    WHEN fqtr = 1 THEN CAST(oancfy AS DOUBLE)
                    WHEN prev_oancfy IS NULL THEN CAST(oancfy AS DOUBLE)
                    ELSE CAST(oancfy AS DOUBLE) - CAST(prev_oancfy AS DOUBLE)
                END as quarterly_ocf,
                CAST(ltq AS DOUBLE) as total_liabilities
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.quarterly_ocf,
            q.total_liabilities,
            -- Calculate Cash Flow to Liability ratio
            q.quarterly_ocf / NULLIF(q.total_liabilities, 0) as cash_flow_to_liability
        FROM temp_universe u
        LEFT JOIN latest_data q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'cash_flow_to_liability']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Cash Flow to Liability ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['cash_flow_to_liability'].notna().sum()
        
        print(f"\nCash Flow to Liability Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid CF/Liability: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nQuarterly Operating Cash Flow Distribution (millions):")
            print(df['quarterly_ocf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTotal Liabilities Distribution (millions):")
            print(df['total_liabilities'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCash Flow to Liability Ratio Distribution:")
            print(df['cash_flow_to_liability'].describe(percentiles=[.05, .25, .5, .75, .95]))


class GrossProfitMarginCalculator(FactorCalculator):
    """
    Calculator for Gross Profit Margin.
    Gross Profit Margin = (Revenue - Cost of Goods Sold) / Revenue
    
    This ratio measures the efficiency of a company's manufacturing operations.
    
    Compustat fields:
    - Revenue: saleq (quarterly sales)
    - Cost of Goods Sold: cogsq
    """
    def __init__(self):
        super().__init__("gross_profit_margin")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(saleq AS DOUBLE) as revenue,
                CAST(cogsq AS DOUBLE) as cogs
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.revenue,
            q.cogs,
            -- Calculate Gross Profit Margin
            CASE 
                WHEN q.revenue <= 0 THEN NULL
                ELSE (q.revenue - q.cogs) / q.revenue
            END as gross_profit_margin
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'gross_profit_margin']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Gross Profit Margin calculation"""
        total_stocks = len(df)
        stocks_with_gpm = df['gross_profit_margin'].notna().sum()
        
        print(f"\nGross Profit Margin Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Gross Profit Margin: {stocks_with_gpm}")
        print(f"Coverage ratio: {(stocks_with_gpm/total_stocks)*100:.2f}%")
        
        if stocks_with_gpm > 0:
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCost of Goods Sold Distribution (millions):")
            print(df['cogs'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nGross Profit Margin Distribution:")
            print(df['gross_profit_margin'].describe(percentiles=[.05, .25, .5, .75, .95]))


class GrowthCalculator(FactorCalculator):
    """
    Base class for calculating growth in financial metrics with flexible lookback periods.
    
    This implementation supports both whole-year lookbacks (4, 8, 12 quarters) as well as
    partial-year lookbacks (e.g., 6 quarters = 1 year and 2 quarters).
    """
    def __init__(self, factor_name, lookback_quarters=4, metric_name="metric", use_pct_change=False):
        super().__init__(factor_name)
        self.lookback_quarters = lookback_quarters
        self.metric_name = metric_name
        self.use_pct_change = use_pct_change
        
        # Calculate years and remaining quarters for lookback
        self.years_back = lookback_quarters // 4
        self.extra_quarters = lookback_quarters % 4
        
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']])
        
        # For the historical period, we adjust by years_back and then adjust the fiscal quarter
        # based on the extra quarters
        query = f"""
        WITH current_data AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as qrank
            FROM wrds_csq_pit c
            INNER JOIN (
                SELECT DISTINCT gvkey FROM temp_universe
            ) u ON c.gvkey = u.gvkey
            WHERE 
                c.datadate <= DATE '{date_str}'
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
            QUALIFY qrank = 1
        ),
        historical_data AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (PARTITION BY c.gvkey, c.fyearq, c.fqtr ORDER BY c.datadate DESC) as qrank
            FROM wrds_csq_pit c
            INNER JOIN current_data cd ON c.gvkey = cd.gvkey 
            WHERE 
                c.datadate <= DATE '{date_str}'
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
                -- First adjust by years back
                AND c.fyearq = cd.fyearq - {self.years_back}
                -- Then adjust fiscal quarter based on extra quarters
                AND c.fqtr = CASE 
                    WHEN cd.fqtr - {self.extra_quarters} <= 0 
                    THEN cd.fqtr - {self.extra_quarters} + 4
                    ELSE cd.fqtr - {self.extra_quarters}
                END
                -- If we need to go back one more year due to quarter adjustment
                AND (
                    (cd.fqtr - {self.extra_quarters} <= 0 AND c.fyearq = cd.fyearq - {self.years_back} - 1)
                    OR
                    (cd.fqtr - {self.extra_quarters} > 0 AND c.fyearq = cd.fyearq - {self.years_back})
                )
            QUALIFY qrank = 1
        ),
        combined_data AS (
            SELECT
                cd.gvkey,
                cd.datadate as current_date,
                cd.fyearq as current_fyear,
                cd.fqtr as current_fqtr,
                hd.datadate as historical_date,
                hd.fyearq as historical_fyear,
                hd.fqtr as historical_fqtr,
                DATEDIFF('day', hd.datadate, cd.datadate) as days_diff,
                {self._get_financial_columns()}
            FROM current_data cd
            LEFT JOIN historical_data hd ON cd.gvkey = hd.gvkey
        )
        SELECT 
            -- Select a single PERMNO record per GVKEY to avoid duplicates
            t.permno, t.lpermno, t.lpermco, t.gvkey, t.iid,
            c.*
        FROM (
            -- For each GVKEY, select only ONE corresponding PERMNO record
            SELECT DISTINCT ON (gvkey) permno, lpermno, lpermco, gvkey, iid
            FROM temp_universe
        ) t
        LEFT JOIN combined_data c ON t.gvkey = c.gvkey
        """
        
        try:
            result_df = duck_conn.execute(query).fetchdf()
            
            # Make doubly sure we don't have duplicates
            if len(result_df) > 0 and 'permno' in result_df.columns:
                result_df = result_df.drop_duplicates(subset=['permno'])
            
            # Calculate metrics
            result_df['current_metric'] = result_df.apply(
                lambda x: self.calculate_metric(x, 'current'), axis=1
            )
            result_df['historical_metric'] = result_df.apply(
                lambda x: self.calculate_metric(x, 'historical'), axis=1
            )
            
            # Calculate growth rate - either as percentage change or absolute difference
            if self.use_pct_change:
                result_df[self.factor_name] = result_df.apply(
                    lambda x: (x['current_metric'] - x['historical_metric'])/abs(x['historical_metric'])
                    if (pd.notnull(x['historical_metric']) and 
                        pd.notnull(x['current_metric']) and 
                        abs(x['historical_metric']) > 1e-6)
                    else None,
                    axis=1
                )
            else:
                # Absolute difference
                result_df[self.factor_name] = result_df.apply(
                    lambda x: (x['current_metric'] - x['historical_metric'])
                    if (pd.notnull(x['historical_metric']) and 
                        pd.notnull(x['current_metric']))
                    else None,
                    axis=1
                )
            
            self._print_summary(result_df, date_str)
            return result_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 
                            'current_date', self.factor_name]].rename(
                                columns={'current_date': 'date'})
            
        finally:
            duck_conn.unregister('temp_universe')

    def _get_financial_columns(self):
        """Return SQL fragment for required financial columns"""
        raise NotImplementedError

    def calculate_metric(self, row, period):
        """Calculate the growth metric for given period (current/historical)"""
        raise NotImplementedError

    def _print_summary(self, df, date_str):
        """Print summary statistics for the calculated growth factor"""
        print(f"\n{'-'*40}")
        if self.extra_quarters == 0:
            period_desc = f"{self.years_back}-Year"
        else:
            period_desc = f"{self.years_back}-Year {self.extra_quarters}-Quarter"
        
        growth_type = "Change" if self.use_pct_change else "Difference"    
        print(f"{self.metric_name.upper()} {period_desc} {growth_type} Summary ({date_str})")
        
        # Count unique PERMNOs
        total_permnos = df['permno'].nunique() if 'permno' in df.columns else 0
        
        # Count PERMNOs with valid growth calculations
        valid_growth_df = df.dropna(subset=[self.factor_name])
        valid_growth = valid_growth_df['permno'].nunique() if 'permno' in valid_growth_df.columns else 0
        
        print(f"Total unique PERMNOs: {total_permnos}")
        print(f"PERMNOs with valid growth calculations: {valid_growth}")
        
        if total_permnos > 0:
            print(f"Coverage ratio: {(valid_growth/total_permnos)*100:.2f}%")
            
            # Check for duplications
            if len(df) > total_permnos:
                print(f"WARNING: {len(df) - total_permnos} duplicate records detected!")
        
        if 'days_diff' in df.columns:
            median_days = df['days_diff'].dropna().median()
            print(f"Median days between periods: {median_days:.1f}")
            expected_days = (self.years_back * 365) + (self.extra_quarters * 91)
            print(f"Expected days between periods: ~{expected_days}")
            
        # Check if we've got matching quarters (for seasonality)
        if 'current_fqtr' in df.columns and 'historical_fqtr' in df.columns:
            # Only count rows that have both valid quarters and valid growth
            valid_quarters_df = valid_growth_df.dropna(subset=['current_fqtr', 'historical_fqtr'])
            matching_quarters = sum(valid_quarters_df['current_fqtr'] == valid_quarters_df['historical_fqtr'])
            quarter_match_pct = (matching_quarters / len(valid_quarters_df)) * 100 if len(valid_quarters_df) > 0 else 0
            print(f"Matching fiscal quarters: {matching_quarters} of {len(valid_quarters_df)} ({quarter_match_pct:.1f}%)")
        
        print(f"\nGrowth {growth_type} distribution:")
        print(df[self.factor_name].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))


class CFOAGrowthCalculator(GrowthCalculator):
    """Cash Flow Over Assets growth with YTD conversion and flexible lookback periods"""
    def __init__(self, lookback_quarters=4, use_pct_change=False):
        super().__init__(
            factor_name=f"cfoa_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="CFOA",
            use_pct_change=use_pct_change
        )

    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']])
        
        # For the historical period, we adjust by years_back and then adjust the fiscal quarter
        # based on the extra quarters
        query = f"""
        WITH current_year_data AS (
            SELECT 
                c.*,
                LAG(c.oancfy) OVER (PARTITION BY c.gvkey ORDER BY c.datadate) as prev_quarter_ytd_ocf,
                ROW_NUMBER() OVER (PARTITION BY c.gvkey ORDER BY c.datadate DESC) as qrank
            FROM wrds_csq_pit c
            INNER JOIN (
                SELECT DISTINCT gvkey FROM temp_universe
            ) u ON c.gvkey = u.gvkey
            WHERE 
                c.datadate <= DATE '{date_str}'
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
            QUALIFY qrank <= 5  -- Get current quarter + previous for YTD calc
        ),
        current_quarter AS (
            SELECT *,
            -- Calculate quarterly OCF from YTD values
            CASE 
                WHEN fqtr = 1 THEN oancfy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_ocf IS NULL THEN oancfy  -- No previous quarter data
                ELSE oancfy - prev_quarter_ytd_ocf  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_ocf
            FROM current_year_data
            WHERE qrank = 1
        ),
        historical_data AS (
            SELECT 
                h.*,
                LAG(h.oancfy) OVER (PARTITION BY h.gvkey ORDER BY h.datadate) as prev_quarter_ytd_ocf,
                ROW_NUMBER() OVER (PARTITION BY h.gvkey, h.fyearq, h.fqtr ORDER BY h.datadate DESC) as qrank
            FROM wrds_csq_pit h
            INNER JOIN current_quarter cd ON h.gvkey = cd.gvkey 
            WHERE 
                h.datadate <= DATE '{date_str}'
                AND h.indfmt = 'INDL'
                AND h.datafmt = 'STD'
                AND h.consol = 'C'
                AND h.popsrc = 'D'
                AND h.curcdq = 'USD'
                AND h.updq = 3
                -- First adjust by years back
                AND (
                    -- If we need to go back one more year due to quarter adjustment
                    (cd.fqtr - {self.extra_quarters} <= 0 AND h.fyearq = cd.fyearq - {self.years_back} - 1)
                    OR
                    -- Standard year adjustment
                    (cd.fqtr - {self.extra_quarters} > 0 AND h.fyearq = cd.fyearq - {self.years_back})
                )
                -- Then adjust fiscal quarter based on extra quarters
                AND h.fqtr = CASE 
                    WHEN cd.fqtr - {self.extra_quarters} <= 0 
                    THEN cd.fqtr - {self.extra_quarters} + 4
                    ELSE cd.fqtr - {self.extra_quarters}
                END
            QUALIFY qrank = 1
        ),
        historical_quarter AS (
            SELECT *,
            -- Calculate quarterly OCF from YTD values
            CASE 
                WHEN fqtr = 1 THEN oancfy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_ocf IS NULL THEN oancfy  -- No previous quarter data
                ELSE oancfy - prev_quarter_ytd_ocf  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_ocf
            FROM historical_data
        ),
        combined_data AS (
            SELECT
                cq.gvkey,
                cq.datadate as current_date,
                cq.fyearq as current_fyear,
                cq.fqtr as current_fqtr,
                hq.datadate as historical_date,
                hq.fyearq as historical_fyear,
                hq.fqtr as historical_fqtr,
                -- Current period CFOA calculation - handle nulls and zeros properly
                CASE
                    WHEN cq.atq IS NULL OR cq.atq = 0 THEN NULL
                    WHEN cq.quarterly_ocf IS NULL THEN NULL
                    ELSE cq.quarterly_ocf / cq.atq
                END as current_cfoa,
                -- Historical period CFOA calculation - handle nulls and zeros properly
                CASE
                    WHEN hq.atq IS NULL OR hq.atq = 0 THEN NULL
                    WHEN hq.quarterly_ocf IS NULL THEN NULL
                    ELSE hq.quarterly_ocf / hq.atq
                END as historical_cfoa,
                DATEDIFF('day', hq.datadate, cq.datadate) as days_diff
            FROM current_quarter cq
            LEFT JOIN historical_quarter hq ON cq.gvkey = hq.gvkey
        )
        SELECT 
            -- Select a single PERMNO record per GVKEY to avoid duplicates
            t.permno, t.lpermno, t.lpermco, t.gvkey, t.iid,
            c.*
        FROM (
            -- For each GVKEY, select only ONE corresponding PERMNO record
            SELECT DISTINCT ON (gvkey) permno, lpermno, lpermco, gvkey, iid
            FROM temp_universe
        ) t
        LEFT JOIN combined_data c ON t.gvkey = c.gvkey
        """
        
        try:
            result_df = duck_conn.execute(query).fetchdf()
            
            # Make doubly sure we don't have duplicates
            if len(result_df) > 0 and 'permno' in result_df.columns:
                result_df = result_df.drop_duplicates(subset=['permno'])
            
            # Calculate growth rate based on choice between percentage or difference
            if self.use_pct_change:
                result_df[self.factor_name] = result_df.apply(
                    lambda x: (x['current_cfoa'] - x['historical_cfoa'])/abs(x['historical_cfoa'])
                    if (pd.notnull(x['historical_cfoa']) and 
                        pd.notnull(x['current_cfoa']) and 
                        abs(x['historical_cfoa']) > 1e-6)
                    else None,
                    axis=1
                )
            else:
                result_df[self.factor_name] = result_df.apply(
                    lambda x: (x['current_cfoa'] - x['historical_cfoa'])
                    if (pd.notnull(x['historical_cfoa']) and 
                        pd.notnull(x['current_cfoa']))
                    else None,
                    axis=1
                )
            
            self._print_summary(result_df, date_str)
            return result_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 
                            'current_date', self.factor_name]].rename(
                                columns={'current_date': 'date'})
        finally:
            duck_conn.unregister('temp_universe')


class GPOAGrowthCalculator(GrowthCalculator):
    """Gross Profit Over Assets growth with flexible lookback periods"""
    def __init__(self, lookback_quarters=4, use_pct_change=False):
        super().__init__(
            factor_name=f"gpoa_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="GPOA",
            use_pct_change=use_pct_change
        )

    def _get_financial_columns(self):
        return """
            cd.saleq as current_sales,
            cd.cogsq as current_cogs,
            cd.atq as current_assets,
            hd.saleq as historical_sales,
            hd.cogsq as historical_cogs,
            hd.atq as historical_assets
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        sales = row.get(f"{prefix}sales")
        cogs = row.get(f"{prefix}cogs")
        assets = row.get(f"{prefix}assets")
        
        if None in [sales, cogs, assets] or assets <= 0:
            return None
        return (sales - cogs) / assets


class ROEGrowthCalculator(GrowthCalculator):
    """Return on Equity growth with flexible lookback periods"""
    def __init__(self, lookback_quarters=4, use_pct_change=False):
        super().__init__(
            factor_name=f"roe_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="ROE",
            use_pct_change=use_pct_change
        )

    def _get_financial_columns(self):
        return """
            cd.ibq as current_income,
            cd.ceqq as current_equity,
            hd.ibq as historical_income,
            hd.ceqq as historical_equity
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        income = row.get(f"{prefix}income")
        equity = row.get(f"{prefix}equity")
        
        if None in [income, equity] or equity <= 0:
            return None
        return income / equity


class ROAGrowthCalculator(GrowthCalculator):
    """Return on Assets growth with flexible lookback periods"""
    def __init__(self, lookback_quarters=4, use_pct_change=False):
        super().__init__(
            factor_name=f"roa_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="ROA",
            use_pct_change=use_pct_change
        )

    def _get_financial_columns(self):
        return """
            cd.ibq as current_income,
            cd.atq as current_assets,
            hd.ibq as historical_income,
            hd.atq as historical_assets
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        income = row.get(f"{prefix}income")
        assets = row.get(f"{prefix}assets")
        
        if None in [income, assets] or assets <= 0:
            return None
        return income / assets


class AccrualsCalculator(FactorCalculator):
    """
    Calculator for Accruals (ACC).
    ACC = ((ΔB - ΔCash) - (ΔA - ΔSTD - ΔTP)) / TA
    
    Where:
    - ΔB: Change in book value
    - ΔCash: Change in cash and short-term investments
    - ΔA: Change in total assets
    - ΔSTD: Change in short-term debt
    - ΔTP: Change in income taxes payable
    - TA: Total assets
    
    According to the paper, ACC is defined as:
    "depreciation minus changes in working capital minus capex, all divided by total assets"
    Working capital (WC) is defined as "current assets minus current liabilities minus cash 
    and short-term instruments plus short-term debt and income taxes payable"
    
    Compustat fields:
    - Depreciation: dpq
    - Current Assets: actq
    - Current Liabilities: lctq
    - Cash and Short-term Investments: cheq
    - Short-term Debt: dlcq
    - Income Taxes Payable: txpq
    - Capital Expenditure: capxy (annual) or capsq (quarterly) if available
    - Total Assets: atq
    """
    def __init__(self):
        super().__init__("acc")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        quarters AS (
            SELECT *
            FROM fundamentals
            WHERE quarter_rank <= 2  -- Current and previous quarter
        ),
        working_capital_calc AS (
            SELECT
                gvkey,
                quarter_rank,
                datadate,
                -- Working Capital = Current Assets - Current Liabilities - Cash + Short-term Debt + Income Taxes Payable
                (CAST(actq AS DOUBLE) - CAST(lctq AS DOUBLE) - CAST(cheq AS DOUBLE) + 
                COALESCE(CAST(dlcq AS DOUBLE), 0) + COALESCE(CAST(txpq AS DOUBLE), 0)) as working_capital,
                CAST(dpq AS DOUBLE) as depreciation,
                CAST(atq AS DOUBLE) as total_assets,
                -- Estimate quarterly capex using available columns (capxy, capsq, etc.)
                COALESCE(
                    CAST(capxy AS DOUBLE) / 4,  -- Use annual capex divided by 4 if available 
                    COALESCE(CAST(capsq AS DOUBLE), 0)  -- Or use quarterly capex if available
                ) as capex
            FROM quarters
        ),
        current_and_prev AS (
            SELECT
                c.gvkey,
                MAX(CASE WHEN c.quarter_rank = 1 THEN c.working_capital END) as current_wc,
                MAX(CASE WHEN c.quarter_rank = 2 THEN c.working_capital END) as prev_wc,
                MAX(CASE WHEN c.quarter_rank = 1 THEN c.depreciation END) as current_dep,
                MAX(CASE WHEN c.quarter_rank = 1 THEN c.capex END) as current_capex,
                MAX(CASE WHEN c.quarter_rank = 1 THEN c.total_assets END) as total_assets
            FROM working_capital_calc c
            GROUP BY c.gvkey
        ),
        accruals_calc AS (
            SELECT
                gvkey,
                current_wc,
                prev_wc,
                current_dep,
                current_capex,
                total_assets,
                -- Calculate change in working capital
                (current_wc - COALESCE(prev_wc, current_wc)) as delta_wc,
                -- Accruals = Depreciation - Change in WC - Capex, all divided by total assets
                CASE
                    WHEN total_assets <= 0 THEN NULL
                    ELSE (current_dep - (current_wc - COALESCE(prev_wc, current_wc)) - current_capex) / total_assets
                END as acc
            FROM current_and_prev
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            a.current_dep as depreciation,
            a.delta_wc as change_in_wc,
            a.current_capex as capex,
            a.total_assets,
            a.acc
        FROM temp_universe u
        LEFT JOIN accruals_calc a ON a.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'acc']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Accruals calculation"""
        total_stocks = len(df)
        stocks_with_acc = df['acc'].notna().sum()
        
        print(f"\nAccruals (ACC) Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid ACC: {stocks_with_acc}")
        print(f"Coverage ratio: {(stocks_with_acc/total_stocks)*100:.2f}%")
        
        if stocks_with_acc > 0:
            print("\nDepreciation Distribution (millions):")
            print(df['depreciation'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nChange in Working Capital Distribution (millions):")
            print(df['change_in_wc'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCapex Distribution (millions):")
            print(df['capex'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTotal Assets Distribution (millions):")
            print(df['total_assets'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nAccruals Distribution:")
            print(df['acc'].describe(percentiles=[.05, .25, .5, .75, .95]))


class WorkingCapitalCalculator(FactorCalculator):
    """
    Calculator for Working Capital (WC).
    WC = Current Assets - Current Liabilities - Cash and Short-term Instruments + Short-term Debt + Income Taxes Payable
    
    This is a key component for calculating accruals and is a measure of a company's operational liquidity.
    
    Compustat fields:
    - Current Assets: actq
    - Current Liabilities: lctq
    - Cash and Short-term Investments: cheq
    - Short-term Debt: dlcq
    - Income Taxes Payable: txpq
    """
    def __init__(self):
        super().__init__("working_capital")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(actq AS DOUBLE) as current_assets,
                CAST(lctq AS DOUBLE) as current_liabilities,
                CAST(cheq AS DOUBLE) as cash_equiv,
                COALESCE(CAST(dlcq AS DOUBLE), 0) as short_term_debt,
                COALESCE(CAST(txpq AS DOUBLE), 0) as income_taxes_payable,
                CAST(atq AS DOUBLE) as total_assets
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.current_assets,
            q.current_liabilities,
            q.cash_equiv,
            q.short_term_debt,
            q.income_taxes_payable,
            q.total_assets,
            -- Calculate Working Capital
            (q.current_assets - q.current_liabilities - q.cash_equiv + q.short_term_debt + q.income_taxes_payable) as working_capital,
            -- Calculate Working Capital as a percentage of Total Assets
            CASE
                WHEN q.total_assets <= 0 THEN NULL
                ELSE (q.current_assets - q.current_liabilities - q.cash_equiv + q.short_term_debt + q.income_taxes_payable) / q.total_assets
            END as working_capital_to_assets
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'working_capital', 'working_capital_to_assets']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Working Capital calculation"""
        total_stocks = len(df)
        stocks_with_wc = df['working_capital'].notna().sum()
        
        print(f"\nWorking Capital Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Working Capital: {stocks_with_wc}")
        print(f"Coverage ratio: {(stocks_with_wc/total_stocks)*100:.2f}%")
        
        if stocks_with_wc > 0:
            print("\nCurrent Assets Distribution (millions):")
            print(df['current_assets'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCurrent Liabilities Distribution (millions):")
            print(df['current_liabilities'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCash and Equivalents Distribution (millions):")
            print(df['cash_equiv'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nWorking Capital Distribution (millions):")
            print(df['working_capital'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nWorking Capital to Assets Ratio Distribution:")
            print(df['working_capital_to_assets'].describe(percentiles=[.05, .25, .5, .75, .95]))


class GPOACalculator(FactorCalculator):
    """
    Calculator for Gross Profits Over Assets (GPOA).
    GPOA = (Revenue - COGS) / Total Assets
    
    This ratio measures how efficiently a company generates gross profits from its assets.
    
    Compustat fields:
    - Revenue: saleq (quarterly sales)
    - COGS: cogsq (cost of goods sold)
    - Total Assets: atq
    """
    def __init__(self):
        super().__init__("gpoa")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(saleq AS DOUBLE) as revenue,
                CAST(cogsq AS DOUBLE) as cogs,
                CAST(atq AS DOUBLE) as total_assets
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.revenue,
            q.cogs,
            q.total_assets,
            (q.revenue - q.cogs) as gross_profit,
            -- Calculate GPOA
            CASE 
                WHEN q.total_assets <= 0 THEN NULL
                ELSE (q.revenue - q.cogs) / q.total_assets
            END as gpoa
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'gpoa']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of GPOA calculation"""
        total_stocks = len(df)
        stocks_with_gpoa = df['gpoa'].notna().sum()
        
        print(f"\nGross Profits Over Assets (GPOA) Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid GPOA: {stocks_with_gpoa}")
        print(f"Coverage ratio: {(stocks_with_gpoa/total_stocks)*100:.2f}%")
        
        if stocks_with_gpoa > 0:
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCOGS Distribution (millions):")
            print(df['cogs'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTotal Assets Distribution (millions):")
            print(df['total_assets'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nGross Profit Distribution (millions):")
            print(df['gross_profit'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nGPOA Distribution:")
            print(df['gpoa'].describe(percentiles=[.05, .25, .5, .75, .95]))


class EarningsToPriceCalculator(FactorCalculator):
    """
    Calculator for Earnings-to-Price ratio (EP).
    EP = Earnings / Market Cap
    
    This is the inverse of the Price-to-Earnings ratio and is a common value metric.
    
    Compustat fields:
    - Earnings: niq (Net Income - Loss)
    - Market Cap: from CRSP (dlycap)
    """
    def __init__(self):
        super().__init__("earnings_to_price")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(niq AS DOUBLE) as net_income
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            e.net_income,
            s.dlycap as market_cap,
            -- Calculate Earnings-to-Price ratio
            CASE 
                WHEN s.dlycap <= 0 THEN NULL
                ELSE e.net_income / s.dlycap
            END as earnings_to_price
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN latest_quarter e ON e.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'earnings_to_price']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Earnings-to-Price ratio calculation"""
        total_stocks = len(df)
        stocks_with_ep = df['earnings_to_price'].notna().sum()
        
        print(f"\nEarnings-to-Price Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid E/P: {stocks_with_ep}")
        print(f"Coverage ratio: {(stocks_with_ep/total_stocks)*100:.2f}%")
        
        if stocks_with_ep > 0:
            print("\nQuarterly Net Income Distribution (millions):")
            print(df['net_income'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nEarnings-to-Price Ratio Distribution:")
            print(df['earnings_to_price'].describe(percentiles=[.05, .25, .5, .75, .95]))


class FreeCashFlowToPriceCalculator(FactorCalculator):
    """
    Calculator for Free Cash Flow to Price ratio (FCF/P).
    FCF/P = Free Cash Flow / Market Cap
    
    Free Cash Flow = Operating Cash Flow - Capital Expenditures
    
    Compustat fields:
    - Operating Cash Flow: oancfy (operating activities - net cash flow)
    - Capital Expenditures: capxy (annual) or capsq (quarterly)
    - Market Cap: from CRSP (dlycap)
    """
    def __init__(self):
        super().__init__("fcf_to_price")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank,
                LAG(c.oancfy) OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate
                ) as prev_oancfy,
                LAG(c.capxy) OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate
                ) as prev_capxy
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_data AS (
            SELECT 
                gvkey,
                datadate,
                fqtr,
                -- Calculate quarterly OCF from YTD values
                CASE 
                    WHEN fqtr = 1 THEN CAST(oancfy AS DOUBLE)  -- Q1: Use YTD directly
                    WHEN prev_oancfy IS NULL THEN CAST(oancfy AS DOUBLE)  -- No previous quarter data
                    ELSE CAST(oancfy AS DOUBLE) - CAST(prev_oancfy AS DOUBLE)  -- Q2-Q4: Current YTD minus previous quarter YTD
                END as quarterly_ocf,
                
                -- Calculate quarterly CAPEX from YTD values
                CASE 
                    WHEN fqtr = 1 THEN CAST(capxy AS DOUBLE)  -- Q1: Use YTD directly
                    WHEN prev_capxy IS NULL THEN CAST(capxy AS DOUBLE)  -- No previous quarter data
                    ELSE CAST(capxy AS DOUBLE) - CAST(prev_capxy AS DOUBLE)  -- Q2-Q4: Current YTD minus previous quarter YTD
                END as quarterly_capex_from_ytd,
                
                -- Also consider explicit quarterly capex (capsq) if available
                CAST(capsq AS DOUBLE) as explicit_quarterly_capex
            FROM fundamentals
            WHERE quarter_rank = 1  -- Only need the most recent quarter now that we calculate YTD differences within the subquery
        ),
        fcf_calc AS (
            SELECT
                gvkey,
                quarterly_ocf,
                -- Choose the most appropriate capex value: explicit quarterly if available, otherwise calculated from YTD
                COALESCE(
                    NULLIF(explicit_quarterly_capex, 0),  -- Use explicit if not null/zero
                    quarterly_capex_from_ytd,  -- Otherwise use calculated from YTD
                    0  -- Default to zero if both are NULL
                ) as quarterly_capex,
                -- Calculate Free Cash Flow
                (quarterly_ocf - COALESCE(
                    NULLIF(explicit_quarterly_capex, 0),
                    quarterly_capex_from_ytd, 
                    0
                )) as quarterly_fcf
            FROM latest_data
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            f.quarterly_ocf,
            f.quarterly_capex,
            f.quarterly_fcf,
            s.dlycap as market_cap,
            -- Calculate FCF to Price ratio
            CASE 
                WHEN s.dlycap <= 0 THEN NULL
                ELSE f.quarterly_fcf / s.dlycap
            END as fcf_to_price
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN fcf_calc f ON f.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'fcf_to_price']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of FCF to Price ratio calculation"""
        total_stocks = len(df)
        stocks_with_fcfp = df['fcf_to_price'].notna().sum()
        
        print(f"\nFree Cash Flow to Price Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid FCF/P: {stocks_with_fcfp}")
        print(f"Coverage ratio: {(stocks_with_fcfp/total_stocks)*100:.2f}%")
        
        if stocks_with_fcfp > 0:
            print("\nQuarterly Operating Cash Flow Distribution (millions):")
            print(df['quarterly_ocf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nQuarterly Capital Expenditures Distribution (millions):")
            print(df['quarterly_capex'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nQuarterly Free Cash Flow Distribution (millions):")
            print(df['quarterly_fcf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nFCF to Price Ratio Distribution:")
            print(df['fcf_to_price'].describe(percentiles=[.05, .25, .5, .75, .95]))


class EBITDAtoEVCalculator(FactorCalculator):
    """
    Calculator for EBITDA to Enterprise Value ratio (EBITDA/EV).
    EBITDA/EV = EBITDA / Enterprise Value
    
    This ratio is often used for company valuation as it removes the impact of
    capital structure, depreciation, and taxes.
    
    EBITDA = Operating Income (oiadpq) + Depreciation and Amortization (dpq)
    Enterprise Value = Market Cap + Net Debt
    
    Compustat fields:
    - Operating Income: oiadpq (Operating Income After Depreciation)
    - Depreciation and Amortization: dpq (Depreciation and Amortization - Total)
    - Net Debt = dlttq (long-term debt) + dlcq (current debt) - cheq (cash and equivalents)
    - Market Cap: from CRSP (dlycap)
    """
    def __init__(self):
        super().__init__("ebitda_to_ev")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(oiadpq AS DOUBLE) as operating_income,
                CAST(dpq AS DOUBLE) as depreciation_amort,
                COALESCE(CAST(dlttq AS DOUBLE), 0) as long_term_debt,
                COALESCE(CAST(dlcq AS DOUBLE), 0) as current_debt,
                COALESCE(CAST(cheq AS DOUBLE), 0) as cash_equivalents
            FROM fundamentals
            WHERE quarter_rank = 1
        ),
        ebitda_calc AS (
            SELECT
                gvkey,
                operating_income,
                depreciation_amort,
                long_term_debt,
                current_debt,
                cash_equivalents,
                -- Calculate quarterly EBITDA
                (operating_income + depreciation_amort) as quarterly_ebitda,
                -- Calculate Net Debt
                (long_term_debt + current_debt - cash_equivalents) as net_debt
            FROM latest_quarter
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            e.quarterly_ebitda,
            s.dlycap as market_cap,
            e.net_debt,
            -- Calculate Enterprise Value
            (s.dlycap + e.net_debt) as enterprise_value,
            -- Calculate EBITDA to EV ratio
            CASE 
                WHEN (s.dlycap + e.net_debt) <= 0 THEN NULL
                ELSE e.quarterly_ebitda / (s.dlycap + e.net_debt)
            END as ebitda_to_ev
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN ebitda_calc e ON e.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'ebitda_to_ev']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of EBITDA to EV ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['ebitda_to_ev'].notna().sum()
        
        print(f"\nEBITDA to Enterprise Value Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid EBITDA/EV: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nQuarterly EBITDA Distribution (millions):")
            print(df['quarterly_ebitda'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nNet Debt Distribution (millions):")
            print(df['net_debt'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nEnterprise Value Distribution (millions):")
            print(df['enterprise_value'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nEBITDA to EV Ratio Distribution:")
            print(df['ebitda_to_ev'].describe(percentiles=[.05, .25, .5, .75, .95]))


class SalesToPriceCalculator(FactorCalculator):
    """
    Calculator for Sales-to-Price ratio (S/P).
    S/P = Revenue / Market Cap
    
    This is the inverse of the Price-to-Sales ratio and is often used as a value metric.
    
    Compustat fields:
    - Revenue: saleq (quarterly sales)
    - Market Cap: from CRSP (dlycap)
    """
    def __init__(self):
        super().__init__("sales_to_price")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(saleq AS DOUBLE) as quarterly_sales
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            s.quarterly_sales,
            m.dlycap as market_cap,
            -- Calculate Sales-to-Price ratio
            CASE 
                WHEN m.dlycap <= 0 THEN NULL
                ELSE s.quarterly_sales / m.dlycap
            END as sales_to_price
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata m ON m.permno = u.permno
            AND m.dlycaldt = DATE '{date}'
        LEFT JOIN latest_quarter s ON s.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'sales_to_price']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Sales-to-Price ratio calculation"""
        total_stocks = len(df)
        stocks_with_sp = df['sales_to_price'].notna().sum()
        
        print(f"\nSales-to-Price Ratio Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid S/P: {stocks_with_sp}")
        print(f"Coverage ratio: {(stocks_with_sp/total_stocks)*100:.2f}%")
        
        if stocks_with_sp > 0:
            print("\nQuarterly Sales Distribution (millions):")
            print(df['quarterly_sales'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nSales-to-Price Ratio Distribution:")
            print(df['sales_to_price'].describe(percentiles=[.05, .25, .5, .75, .95]))

class CumulativeReturnCalculator(FactorCalculator):
    """
    Calculator for cumulative returns over a specified lookback period.
    
    This calculator computes the raw cumulative return for each stock
    over a specified number of days prior to a given end date.
    
    Parameters:
    - lookback_days: Number of days to look back for cumulative return calculation
    - end_date_offset: Number of days to offset from the calculation date (0 means use the calculation date)
    """
    def __init__(self, lookback_days, end_date_offset=0):
        # Initialize with descriptive factor name that includes both lookback period and offset
        if end_date_offset == 0:
            factor_name = f"cum_return_{lookback_days}d"
        else:
            factor_name = f"cum_return_{lookback_days}d_offset_{end_date_offset}d"
        super().__init__(factor_name)
        self.lookback_days = lookback_days
        self.end_date_offset = end_date_offset
        self.factor_name = factor_name
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH date_ranges AS (
            SELECT 
                DATE '{date}' - INTERVAL '{end_offset} days' as end_date,
                DATE '{date}' - INTERVAL '{end_offset} days' - INTERVAL '{lookback} days' as start_date
        ),
        daily_returns AS (
            SELECT 
                s.permno,
                s.dlycaldt,
                s.DlyRet
            FROM stkdlysecuritydata s
            INNER JOIN temp_universe u ON u.permno = s.permno
            INNER JOIN date_ranges d 
                ON s.dlycaldt <= d.end_date
                AND s.dlycaldt > d.start_date
            WHERE s.DlyRet IS NOT NULL
        ),
        cumulative_returns AS (
            SELECT 
                permno,
                COUNT(*) as num_observations,
                -- Use product aggregation for returns (to handle potential -1 values)
                (PRODUCT(1 + CAST(DlyRet AS DOUBLE)) - 1) as {factor_name}
            FROM daily_returns
            GROUP BY permno
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            cr.{factor_name},
            cr.num_observations
        FROM temp_universe u
        LEFT JOIN cumulative_returns cr ON cr.permno = u.permno
        """
        
        try:
            result = duck_conn.execute(
                query.format(
                    date=date_str,
                    lookback=self.lookback_days,
                    end_offset=self.end_date_offset,
                    factor_name=self.factor_name
                )
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            # Remove the num_observations column for the final result
            final_result = result.drop(['num_observations'], axis=1)
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        total_stocks = len(df)
        stocks_with_factor = df[self.factor_name].notna().sum()
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Total stocks in universe: {total_stocks}")
        print(f"Stocks with valid cumulative returns: {stocks_with_factor}")
        print(f"Coverage ratio: {(stocks_with_factor/total_stocks)*100:.2f}%")
        
        if stocks_with_factor > 0:
            print("\nCumulative Return Distribution:")
            print(df[self.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            # Add additional statistics about observation count
            if 'num_observations' in df.columns:
                print("\nNumber of Trading Days in Period:")
                print(df['num_observations'].describe(percentiles=[.05, .25, .5, .75, .95]))

class EBITDAtoTEVCalculator(FactorCalculator):
    """
    Calculator for EBITDA to Total Enterprise Value ratio (Enterprise Yield).
    
    Enterprise Yield = EBITDA / TEV
    
    Where:
    - EBITDA = Earnings Before Interest, Taxes, Depreciation, and Amortization
    - TEV = Market Cap + Total Debt - Excess Cash + Preferred Stock + Minority Interests
    - Excess Cash = Cash + Current Assets - Current Liabilities
    
    This measure reflects the true cost of total acquisition and shows the unadulterated 
    operating earnings flowing to the acquirer post-acquisition.
    
    Compustat fields:
    - Operating Income: oiadpq (Operating Income After Depreciation)
    - Depreciation and Amortization: dpq (Depreciation and Amortization - Total)
    - Total Debt: dlttq (Long-term Debt) + dlcq (Current Debt)
    - Cash: cheq (Cash and Short-Term Investments)
    - Current Assets: actq
    - Current Liabilities: lctq
    - Preferred Stock: pstkq (Preferred/Preference Stock - Total)
    - Minority Interests: mibtq (Noncontrolling Interests - Total - Balance Sheet)
    - Market Cap: from CRSP (dlycap)
    """
    def __init__(self):
        super().__init__("ebitda_to_tev")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(oiadpq AS DOUBLE) as operating_income,
                CAST(dpq AS DOUBLE) as depreciation_amort,
                COALESCE(CAST(dlttq AS DOUBLE), 0) as long_term_debt,
                COALESCE(CAST(dlcq AS DOUBLE), 0) as current_debt,
                COALESCE(CAST(cheq AS DOUBLE), 0) as cash_equivalents,
                COALESCE(CAST(actq AS DOUBLE), 0) as current_assets,
                COALESCE(CAST(lctq AS DOUBLE), 0) as current_liabilities,
                COALESCE(CAST(pstkq AS DOUBLE), 0) as preferred_stock,
                COALESCE(CAST(mibtq AS DOUBLE), 0) as minority_interests
            FROM fundamentals
            WHERE quarter_rank = 1
        ),
        tev_calc AS (
            SELECT
                gvkey,
                operating_income,
                depreciation_amort,
                long_term_debt,
                current_debt,
                cash_equivalents,
                current_assets,
                current_liabilities,
                preferred_stock,
                minority_interests,
                -- Calculate EBITDA
                (operating_income + depreciation_amort) as ebitda,
                -- Calculate total debt
                (long_term_debt + current_debt) as total_debt,
                -- Calculate excess cash according to definition
                (cash_equivalents + current_assets - current_liabilities) as excess_cash
            FROM latest_quarter
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            t.ebitda,
            t.total_debt,
            t.excess_cash,
            t.preferred_stock,
            t.minority_interests,
            s.dlycap as market_cap,
            -- Calculate TEV
            (s.dlycap + t.total_debt - t.excess_cash + t.preferred_stock + t.minority_interests) as tev,
            -- Calculate EBITDA to TEV ratio (Enterprise Yield)
            CASE 
                WHEN (s.dlycap + t.total_debt - t.excess_cash + t.preferred_stock + t.minority_interests) <= 0 THEN NULL
                ELSE t.ebitda / (s.dlycap + t.total_debt - t.excess_cash + t.preferred_stock + t.minority_interests)
            END as ebitda_to_tev
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN tev_calc t ON t.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'ebitda_to_tev']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of EBITDA to TEV ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['ebitda_to_tev'].notna().sum()
        
        print(f"\nEBITDA to TEV Ratio (Enterprise Yield) Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid EBITDA/TEV: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nEBITDA Distribution (millions):")
            print(df['ebitda'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTotal Debt Distribution (millions):")
            print(df['total_debt'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nExcess Cash Distribution (millions):")
            print(df['excess_cash'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTEV Distribution (millions):")
            print(df['tev'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nEBITDA to TEV Ratio (Enterprise Yield) Distribution:")
            print(df['ebitda_to_tev'].describe(percentiles=[.05, .25, .5, .75, .95]))



class FCFtoTEVCalculator(FactorCalculator):
    """
    Calculator for Free Cash Flow Yield (FCF/TEV).
    
    Free Cash Flow Yield = FCF / TEV
    
    Where:
    - FCF = Net Income + Depreciation and Amortization - Working Capital Change - Capital Expenditures
    - TEV = Market Cap + Total Debt - Excess Cash + Preferred Stock + Minority Interests
    - Excess Cash = Cash + Current Assets - Current Liabilities
    
    Compustat fields:
    - Net Income: niq (Net Income (Loss))
    - Depreciation and Amortization: dpq (Depreciation and Amortization - Total)
    - Working Capital Change: Calculated from current period and previous period wcapq (Working Capital)
    - Capital Expenditures: capxy (Year-to-date capital expenditures)
    - Market Cap: from CRSP (dlycap)
    - Total Debt: dlttq (Long-term Debt) + dlcq (Current Debt)
    - Cash: cheq (Cash and Short-Term Investments)
    - Current Assets: actq
    - Current Liabilities: lctq
    - Preferred Stock: pstkq (Preferred/Preference Stock - Total)
    - Minority Interests: mibtq (Noncontrolling Interests - Total - Balance Sheet)
    """
    def __init__(self):
        super().__init__("fcf_to_tev")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        quarters AS (
            SELECT *,
            -- Track previous period capxy within the same fiscal year
            LAG(capxy) OVER (
                PARTITION BY gvkey, fyearq 
                ORDER BY datadate
            ) as prev_capxy_same_year
            FROM fundamentals
            WHERE quarter_rank <= 2  -- Get current and previous quarter
        ),
        financial_data AS (
            SELECT
                gvkey,
                quarter_rank,
                datadate,
                fyearq,
                fqtr,
                CAST(wcapq AS DOUBLE) as working_capital,
                CAST(niq AS DOUBLE) as net_income,
                CAST(dpq AS DOUBLE) as depreciation_amort,
                CAST(capxy AS DOUBLE) as capxy_ytd,
                prev_capxy_same_year,
                COALESCE(CAST(dlttq AS DOUBLE), 0) as long_term_debt,
                COALESCE(CAST(dlcq AS DOUBLE), 0) as current_debt,
                COALESCE(CAST(cheq AS DOUBLE), 0) as cash_equivalents,
                COALESCE(CAST(actq AS DOUBLE), 0) as current_assets,
                COALESCE(CAST(lctq AS DOUBLE), 0) as current_liabilities,
                COALESCE(CAST(pstkq AS DOUBLE), 0) as preferred_stock,
                COALESCE(CAST(mibtq AS DOUBLE), 0) as minority_interests
            FROM quarters
        ),
        fcf_calc AS (
            SELECT
                curr.gvkey,
                curr.net_income,
                curr.depreciation_amort,
                curr.long_term_debt,
                curr.current_debt,
                curr.cash_equivalents,
                curr.current_assets,
                curr.current_liabilities,
                curr.preferred_stock,
                curr.minority_interests,
                -- Calculate working capital change
                (curr.working_capital - COALESCE(prev.working_capital, curr.working_capital)) as wc_change,
                -- Calculate quarterly capital expenditure based on fiscal quarter
                CASE
                    -- First fiscal quarter: use the YTD value directly
                    WHEN curr.fqtr = 1 THEN curr.capxy_ytd
                    -- Other quarters: calculate the quarterly capxy by taking the difference
                    -- between current YTD and previous quarter YTD in the same fiscal year
                    ELSE curr.capxy_ytd - COALESCE(curr.prev_capxy_same_year, 0)
                END as quarterly_capex,
                -- Calculate total debt
                (curr.long_term_debt + curr.current_debt) as total_debt,
                -- Calculate excess cash according to definition
                (curr.cash_equivalents + curr.current_assets - curr.current_liabilities) as excess_cash
            FROM 
                (SELECT * FROM financial_data WHERE quarter_rank = 1) curr
            LEFT JOIN 
                (SELECT * FROM financial_data WHERE quarter_rank = 2) prev
                ON curr.gvkey = prev.gvkey
        ),
        fcf_final AS (
            SELECT
                gvkey,
                net_income,
                depreciation_amort,
                wc_change,
                quarterly_capex,
                total_debt,
                excess_cash,
                preferred_stock,
                minority_interests,
                -- Calculate Free Cash Flow
                (net_income + depreciation_amort - wc_change - quarterly_capex) as fcf
            FROM fcf_calc
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            f.net_income,
            f.depreciation_amort,
            f.wc_change,
            f.quarterly_capex,
            f.fcf,
            f.total_debt,
            f.excess_cash,
            f.preferred_stock,
            f.minority_interests,
            s.dlycap as market_cap,
            -- Calculate TEV
            (s.dlycap + f.total_debt - f.excess_cash + f.preferred_stock + f.minority_interests) as tev,
            -- Calculate FCF to TEV ratio
            CASE 
                WHEN (s.dlycap + f.total_debt - f.excess_cash + f.preferred_stock + f.minority_interests) <= 0 THEN NULL
                ELSE f.fcf / (s.dlycap + f.total_debt - f.excess_cash + f.preferred_stock + f.minority_interests)
            END as fcf_to_tev
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN fcf_final f ON f.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'fcf_to_tev']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of FCF to TEV ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['fcf_to_tev'].notna().sum()
        
        print(f"\nFree Cash Flow Yield (FCF/TEV) Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid FCF/TEV: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nFCF Components:")
            print("\nNet Income Distribution (millions):")
            print(df['net_income'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nDepreciation & Amortization Distribution (millions):")
            print(df['depreciation_amort'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nWorking Capital Change Distribution (millions):")
            print(df['wc_change'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nQuarterly Capital Expenditures Distribution (millions):")
            print(df['quarterly_capex'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nFree Cash Flow Distribution (millions):")
            print(df['fcf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTEV Distribution (millions):")
            print(df['tev'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nFCF to TEV Ratio (Free Cash Flow Yield) Distribution:")
            print(df['fcf_to_tev'].describe(percentiles=[.05, .25, .5, .75, .95]))

class GrossProfitsToTEVCalculator(FactorCalculator):
    """
    Calculator for Gross Profits Yield (GP/TEV).
    
    Gross Profits Yield = GP / TEV
    
    Where:
    - GP = Revenue - Cost of Goods Sold
    - TEV = Market Cap + Total Debt - Excess Cash + Preferred Stock + Minority Interests
    - Excess Cash = Cash + Current Assets - Current Liabilities
    
    Gross Profits Yield measures the ratio of a company's gross profits to its total enterprise value.
    It is a variation of the enterprise multiple that uses gross profitability instead of EBITDA,
    focusing on the raw profit flowing back to the stock after cost of goods sold is deducted from sales.
    
    Compustat fields:
    - Revenue: revtq or saleq (Revenue/Sales Total)
    - Cost of Goods Sold: cogsq
    - Market Cap: from CRSP (dlycap)
    - Total Debt: dlttq (Long-term Debt) + dlcq (Current Debt)
    - Cash: cheq (Cash and Short-Term Investments)
    - Current Assets: actq
    - Current Liabilities: lctq
    - Preferred Stock: pstkq (Preferred/Preference Stock - Total)
    - Minority Interests: mibtq (Noncontrolling Interests - Total - Balance Sheet)
    """
    def __init__(self):
        super().__init__("gross_profits_to_tev")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                COALESCE(CAST(revtq AS DOUBLE), CAST(saleq AS DOUBLE)) as revenue,
                CAST(cogsq AS DOUBLE) as cogs,
                COALESCE(CAST(dlttq AS DOUBLE), 0) as long_term_debt,
                COALESCE(CAST(dlcq AS DOUBLE), 0) as current_debt,
                COALESCE(CAST(cheq AS DOUBLE), 0) as cash_equivalents,
                COALESCE(CAST(actq AS DOUBLE), 0) as current_assets,
                COALESCE(CAST(lctq AS DOUBLE), 0) as current_liabilities,
                COALESCE(CAST(pstkq AS DOUBLE), 0) as preferred_stock,
                COALESCE(CAST(mibtq AS DOUBLE), 0) as minority_interests
            FROM fundamentals
            WHERE quarter_rank = 1
        ),
        gp_calc AS (
            SELECT
                gvkey,
                revenue,
                cogs,
                long_term_debt,
                current_debt,
                cash_equivalents,
                current_assets,
                current_liabilities,
                preferred_stock,
                minority_interests,
                -- Calculate Gross Profits
                (revenue - cogs) as gross_profits,
                -- Calculate total debt
                (long_term_debt + current_debt) as total_debt,
                -- Calculate excess cash according to definition
                (cash_equivalents + current_assets - current_liabilities) as excess_cash
            FROM latest_quarter
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            g.revenue,
            g.cogs,
            g.gross_profits,
            g.total_debt,
            g.excess_cash,
            g.preferred_stock,
            g.minority_interests,
            s.dlycap as market_cap,
            -- Calculate TEV
            (s.dlycap + g.total_debt - g.excess_cash + g.preferred_stock + g.minority_interests) as tev,
            -- Calculate Gross Profits to TEV ratio
            CASE 
                WHEN (s.dlycap + g.total_debt - g.excess_cash + g.preferred_stock + g.minority_interests) <= 0 THEN NULL
                ELSE g.gross_profits / (s.dlycap + g.total_debt - g.excess_cash + g.preferred_stock + g.minority_interests)
            END as gross_profits_to_tev
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s ON s.permno = u.permno
            AND s.dlycaldt = DATE '{date}'
        LEFT JOIN gp_calc g ON g.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'gross_profits_to_tev']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Gross Profits to TEV ratio calculation"""
        total_stocks = len(df)
        stocks_with_ratio = df['gross_profits_to_tev'].notna().sum()
        
        print(f"\nGross Profits Yield (GP/TEV) Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid GP/TEV: {stocks_with_ratio}")
        print(f"Coverage ratio: {(stocks_with_ratio/total_stocks)*100:.2f}%")
        
        if stocks_with_ratio > 0:
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCost of Goods Sold Distribution (millions):")
            print(df['cogs'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nGross Profits Distribution (millions):")
            print(df['gross_profits'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nMarket Cap Distribution (millions):")
            print(df['market_cap'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nTEV Distribution (millions):")
            print(df['tev'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nGross Profits to TEV Ratio (Gross Profits Yield) Distribution:")
            print(df['gross_profits_to_tev'].describe(percentiles=[.05, .25, .5, .75, .95]))

class OperatingMarginCalculator(FactorCalculator):
    """
    Calculator for Operating Margin.
    Operating Margin = Operating Income / Sales
    
    This ratio measures a company's profitability from its core operations.
    
    Compustat fields:
    - Operating Income: oiadpq (Operating Income After Depreciation)
    - Sales: saleq (quarterly sales)
    """
    def __init__(self):
        super().__init__("operating_margin")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(oiadpq AS DOUBLE) as operating_income,
                CAST(saleq AS DOUBLE) as revenue
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.operating_income,
            q.revenue,
            -- Calculate Operating Margin
            CASE 
                WHEN q.revenue <= 0 THEN NULL
                ELSE q.operating_income / q.revenue
            END as operating_margin
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'operating_margin']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Operating Margin calculation"""
        total_stocks = len(df)
        stocks_with_margin = df['operating_margin'].notna().sum()
        
        print(f"\nOperating Margin Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Operating Margin: {stocks_with_margin}")
        print(f"Coverage ratio: {(stocks_with_margin/total_stocks)*100:.2f}%")
        
        if stocks_with_margin > 0:
            print("\nOperating Income Distribution (millions):")
            print(df['operating_income'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nOperating Margin Distribution:")
            print(df['operating_margin'].describe(percentiles=[.05, .25, .5, .75, .95]))


class NetMarginCalculator(FactorCalculator):
    """
    Calculator for Net Margin (also known as Net Profit Margin).
    Net Margin = Net Income / Sales
    
    This ratio measures a company's overall profitability after all expenses.
    
    Compustat fields:
    - Net Income: niq (Net Income)
    - Sales: saleq (quarterly sales)
    """
    def __init__(self):
        super().__init__("net_margin")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(niq AS DOUBLE) as net_income,
                CAST(saleq AS DOUBLE) as revenue
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.net_income,
            q.revenue,
            -- Calculate Net Margin
            CASE 
                WHEN q.revenue <= 0 THEN NULL
                ELSE q.net_income / q.revenue
            END as net_margin
        FROM temp_universe u
        LEFT JOIN latest_quarter q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'net_margin']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Net Margin calculation"""
        total_stocks = len(df)
        stocks_with_margin = df['net_margin'].notna().sum()
        
        print(f"\nNet Margin Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Net Margin: {stocks_with_margin}")
        print(f"Coverage ratio: {(stocks_with_margin/total_stocks)*100:.2f}%")
        
        if stocks_with_margin > 0:
            print("\nNet Income Distribution (millions):")
            print(df['net_income'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nNet Margin Distribution:")
            print(df['net_margin'].describe(percentiles=[.05, .25, .5, .75, .95]))


class OperatingCashFlowMarginCalculator(FactorCalculator):
    """
    Calculator for Cash Flow Margin.
    Cash Flow Margin = Operating Cash Flow / Sales
    
    This ratio measures a company's ability to convert sales into cash.
    
    Compustat fields:
    - Operating Cash Flow: oancfy (operating activities - net cash flow)
    - Sales: saleq (quarterly sales)
    """
    def __init__(self):
        super().__init__("cash_flow_margin")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank,
                LAG(oancfy) OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate
                ) as prev_oancfy
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_data AS (
            SELECT 
                gvkey,
                CASE 
                    WHEN fqtr = 1 THEN CAST(oancfy AS DOUBLE)
                    WHEN prev_oancfy IS NULL THEN CAST(oancfy AS DOUBLE)
                    ELSE CAST(oancfy AS DOUBLE) - CAST(prev_oancfy AS DOUBLE)
                END as quarterly_ocf,
                CAST(saleq AS DOUBLE) as revenue
            FROM fundamentals
            WHERE quarter_rank = 1
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            q.quarterly_ocf,
            q.revenue,
            -- Calculate Cash Flow Margin
            CASE 
                WHEN q.revenue <= 0 THEN NULL
                ELSE q.quarterly_ocf / q.revenue
            END as cash_flow_margin
        FROM temp_universe u
        LEFT JOIN latest_data q ON q.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'cash_flow_margin']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Cash Flow Margin calculation"""
        total_stocks = len(df)
        stocks_with_margin = df['cash_flow_margin'].notna().sum()
        
        print(f"\nCash Flow Margin Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid Cash Flow Margin: {stocks_with_margin}")
        print(f"Coverage ratio: {(stocks_with_margin/total_stocks)*100:.2f}%")
        
        if stocks_with_margin > 0:
            print("\nQuarterly Operating Cash Flow Distribution (millions):")
            print(df['quarterly_ocf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nCash Flow Margin Distribution:")
            print(df['cash_flow_margin'].describe(percentiles=[.05, .25, .5, .75, .95]))


class FreeCashFlowMarginCalculator(FactorCalculator):
    """
    Calculator for Free Cash Flow Margin.
    Free Cash Flow Margin = Free Cash Flow / Sales
    
    This ratio measures a company's ability to generate free cash flow from sales
    after accounting for operating expenses and capital expenditures.
    
    Free Cash Flow = Operating Cash Flow - Capital Expenditures
    
    Compustat fields:
    - Operating Cash Flow: oancfy (operating activities - net cash flow)
    - Capital Expenditures: capxy (annual) or capsq (quarterly)
    - Sales: saleq (quarterly sales)
    """
    def __init__(self):
        super().__init__("free_cash_flow_margin")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank,
                LAG(c.oancfy) OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate
                ) as prev_oancfy,
                LAG(c.capxy) OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate
                ) as prev_capxy
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_data AS (
            SELECT 
                gvkey,
                datadate,
                fqtr,
                -- Calculate quarterly OCF from YTD values
                CASE 
                    WHEN fqtr = 1 THEN CAST(oancfy AS DOUBLE)  -- Q1: Use YTD directly
                    WHEN prev_oancfy IS NULL THEN CAST(oancfy AS DOUBLE)  -- No previous quarter data
                    ELSE CAST(oancfy AS DOUBLE) - CAST(prev_oancfy AS DOUBLE)  -- Q2-Q4: Current YTD minus previous quarter YTD
                END as quarterly_ocf,
                
                -- Calculate quarterly CAPEX from YTD values
                CASE 
                    WHEN fqtr = 1 THEN CAST(capxy AS DOUBLE)  -- Q1: Use YTD directly
                    WHEN prev_capxy IS NULL THEN CAST(capxy AS DOUBLE)  -- No previous quarter data
                    ELSE CAST(capxy AS DOUBLE) - CAST(prev_capxy AS DOUBLE)  -- Q2-Q4: Current YTD minus previous quarter YTD
                END as quarterly_capex_from_ytd,
                
                -- Also consider explicit quarterly capex (capsq) if available
                CAST(capsq AS DOUBLE) as explicit_quarterly_capex,
                
                CAST(saleq AS DOUBLE) as revenue
            FROM fundamentals
            WHERE quarter_rank = 1  -- Only need the most recent quarter now that we calculate YTD differences within the subquery
        ),
        fcf_calc AS (
            SELECT
                gvkey,
                revenue,
                quarterly_ocf,
                -- Choose the most appropriate capex value: explicit quarterly if available, otherwise calculated from YTD
                COALESCE(
                    NULLIF(explicit_quarterly_capex, 0),  -- Use explicit if not null/zero
                    quarterly_capex_from_ytd,  -- Otherwise use calculated from YTD
                    0  -- Default to zero if both are NULL
                ) as quarterly_capex,
                -- Calculate Free Cash Flow
                (quarterly_ocf - COALESCE(
                    NULLIF(explicit_quarterly_capex, 0),
                    quarterly_capex_from_ytd, 
                    0
                )) as quarterly_fcf
            FROM latest_data
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            f.quarterly_ocf,
            f.quarterly_capex,
            f.quarterly_fcf,
            f.revenue,
            -- Calculate Free Cash Flow Margin
            CASE 
                WHEN f.revenue <= 0 THEN NULL
                ELSE f.quarterly_fcf / f.revenue
            END as free_cash_flow_margin
        FROM temp_universe u
        LEFT JOIN fcf_calc f ON f.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'free_cash_flow_margin']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of Free Cash Flow Margin calculation"""
        total_stocks = len(df)
        stocks_with_margin = df['free_cash_flow_margin'].notna().sum()
        
        print(f"\nFree Cash Flow Margin Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid FCF Margin: {stocks_with_margin}")
        print(f"Coverage ratio: {(stocks_with_margin/total_stocks)*100:.2f}%")
        
        if stocks_with_margin > 0:
            print("\nQuarterly Operating Cash Flow Distribution (millions):")
            print(df['quarterly_ocf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nQuarterly Capital Expenditures Distribution (millions):")
            print(df['quarterly_capex'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nQuarterly Free Cash Flow Distribution (millions):")
            print(df['quarterly_fcf'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nFree Cash Flow Margin Distribution:")
            print(df['free_cash_flow_margin'].describe(percentiles=[.05, .25, .5, .75, .95]))


class EBITDAMarginCalculator(FactorCalculator):
    """
    Calculator for EBITDA Margin.
    EBITDA Margin = EBITDA / Sales
    
    This ratio measures a company's operating profitability before non-operating expenses.
    EBITDA removes the impact of financial decisions, accounting decisions, and tax environments.
    
    EBITDA = Operating Income + Depreciation and Amortization
    
    Compustat fields:
    - Operating Income: oiadpq (Operating Income After Depreciation)
    - Depreciation and Amortization: dpq (Depreciation and Amortization - Total)
    - Sales: saleq (quarterly sales)
    """
    def __init__(self):
        super().__init__("ebitda_margin")
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        WITH fundamentals AS (
            SELECT 
                c.*,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey 
                    ORDER BY c.datadate DESC
                ) as quarter_rank
            FROM wrds_csq_pit c
            WHERE 
                c.pitdate1 <= DATE '{date}'
                AND (c.pitdate2 >= DATE '{date}' OR c.pitdate2 IS NULL)
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
        ),
        latest_quarter AS (
            SELECT 
                gvkey,
                CAST(oiadpq AS DOUBLE) as operating_income,
                CAST(dpq AS DOUBLE) as depreciation_amort,
                CAST(saleq AS DOUBLE) as revenue
            FROM fundamentals
            WHERE quarter_rank = 1
        ),
        ebitda_calc AS (
            SELECT
                gvkey,
                operating_income,
                depreciation_amort,
                revenue,
                -- Calculate EBITDA
                (operating_income + depreciation_amort) as ebitda
            FROM latest_quarter
        )
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            e.operating_income,
            e.depreciation_amort,
            e.ebitda,
            e.revenue,
            -- Calculate EBITDA Margin
            CASE 
                WHEN e.revenue <= 0 THEN NULL
                ELSE e.ebitda / e.revenue
            END as ebitda_margin
        FROM temp_universe u
        LEFT JOIN ebitda_calc e ON e.gvkey = u.gvkey
        """
        
        try:
            result = duck_conn.execute(
                query.format(date=date_str)
            ).fetchdf()
            
            self._print_summary_statistics(result, date_str)
            
            final_columns = ['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 'date', 'ebitda_margin']
            final_result = result[final_columns]
            
            return final_result
            
        finally:
            duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        """Print analysis of EBITDA Margin calculation"""
        total_stocks = len(df)
        stocks_with_margin = df['ebitda_margin'].notna().sum()
        
        print(f"\nEBITDA Margin Calculation Summary for {date_str}")
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid EBITDA Margin: {stocks_with_margin}")
        print(f"Coverage ratio: {(stocks_with_margin/total_stocks)*100:.2f}%")
        
        if stocks_with_margin > 0:
            print("\nOperating Income Distribution (millions):")
            print(df['operating_income'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nDepreciation & Amortization Distribution (millions):")
            print(df['depreciation_amort'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nEBITDA Distribution (millions):")
            print(df['ebitda'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nRevenue Distribution (millions):")
            print(df['revenue'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nEBITDA Margin Distribution:")
            print(df['ebitda_margin'].describe(percentiles=[.05, .25, .5, .75, .95]))


class AssetTurnoverChangeCalculator(GrowthCalculator):
    """
    Calculator for Change in Asset Turnover with flexible lookback periods.
    
    Change in Asset Turnover = ln(Asset Turnover_t) - ln(Asset Turnover_t-1)
    
    This measures the log difference in asset turnover compared to a previous period,
    which approximates the percentage change in asset turnover.
    """
    def __init__(self, lookback_quarters=4):
        super().__init__(
            factor_name=f"asset_turnover_{lookback_quarters}q_change",
            lookback_quarters=lookback_quarters,
            metric_name="Asset Turnover",
            use_pct_change=False  # We'll handle the log difference manually
        )
    
    def _get_financial_columns(self):
        return """
            cd.saleq as current_sales,
            cd.atq as current_assets,
            hd.saleq as historical_sales,
            hd.atq as historical_assets
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        sales = row.get(f"{prefix}sales")
        assets = row.get(f"{prefix}assets")
        
        if None in [sales, assets] or pd.isna(sales) or pd.isna(assets) or assets <= 0:
            return None
        
        # Return the natural log of asset turnover
        return sales / assets
    

class SalesGrowthCalculator(GrowthCalculator):
    """
    Calculator for Sales Growth with flexible lookback periods.
    
    Sales Growth = (Sales_t - Sales_t-1) / Sales_t-1
    
    This measures the percentage change in sales compared to a previous period.
    """
    def __init__(self, lookback_quarters=4):
        super().__init__(
            factor_name=f"sales_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="Sales",
            use_pct_change=True
        )
    
    def _get_financial_columns(self):
        return """
            cd.saleq as current_sales,
            hd.saleq as historical_sales
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        sales = row.get(f"{prefix}sales")
        
        if sales is None or pd.isna(sales) or sales <= 0:
            return None
        return sales
    

class GrossMarginChangeCalculator(GrowthCalculator):
    """
    Calculator for Change in Gross Margin with flexible lookback periods.
    
    Change in Gross Margin = Gross Margin_t - Gross Margin_t-1
    
    This measures the absolute change in gross margin compared to a previous period.
    """
    def __init__(self, lookback_quarters=4):
        super().__init__(
            factor_name=f"gross_margin_{lookback_quarters}q_change",
            lookback_quarters=lookback_quarters,
            metric_name="Gross Margin",
            use_pct_change=False  # Use absolute difference
        )
    
    def _get_financial_columns(self):
        return """
            cd.saleq as current_sales,
            cd.cogsq as current_cogs,
            hd.saleq as historical_sales,
            hd.cogsq as historical_cogs
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        sales = row.get(f"{prefix}sales")
        cogs = row.get(f"{prefix}cogs")
        
        if None in [sales, cogs] or pd.isna(sales) or pd.isna(cogs) or sales <= 0:
            return None
        
        # Calculate gross margin (gross profit / sales)
        return (sales - cogs) / sales


class CashFlowMarginChangeCalculator(GrowthCalculator):
    """
    Calculator for Change in Cash Flow Margin with flexible lookback periods.
    
    Change in Cash Flow Margin = Cash Flow Margin_t - Cash Flow Margin_t-1
    
    This measures the absolute change in cash flow margin compared to a previous period.
    """
    def __init__(self, lookback_quarters=4):
        super().__init__(
            factor_name=f"cash_flow_margin_{lookback_quarters}q_change",
            lookback_quarters=lookback_quarters,
            metric_name="Cash Flow Margin",
            use_pct_change=False  # Use absolute difference
        )
    
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']])
        
        query = f"""
        WITH current_year_data AS (
            SELECT 
                c.*,
                LAG(c.oancfy) OVER (PARTITION BY c.gvkey ORDER BY c.datadate) as prev_quarter_ytd_ocf,
                ROW_NUMBER() OVER (PARTITION BY c.gvkey ORDER BY c.datadate DESC) as qrank
            FROM wrds_csq_pit c
            INNER JOIN (
                SELECT DISTINCT gvkey FROM temp_universe
            ) u ON c.gvkey = u.gvkey
            WHERE 
                c.datadate <= DATE '{date_str}'
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
            QUALIFY qrank <= 5  -- Get current quarter + previous for YTD calc
        ),
        current_quarter AS (
            SELECT *,
            -- Calculate quarterly OCF from YTD values
            CASE 
                WHEN fqtr = 1 THEN oancfy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_ocf IS NULL THEN oancfy  -- No previous quarter data
                ELSE oancfy - prev_quarter_ytd_ocf  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_ocf
            FROM current_year_data
            WHERE qrank = 1
        ),
        historical_data AS (
            SELECT 
                h.*,
                LAG(h.oancfy) OVER (PARTITION BY h.gvkey ORDER BY h.datadate) as prev_quarter_ytd_ocf,
                ROW_NUMBER() OVER (PARTITION BY h.gvkey, h.fyearq, h.fqtr ORDER BY h.datadate DESC) as qrank
            FROM wrds_csq_pit h
            INNER JOIN current_quarter cd ON h.gvkey = cd.gvkey 
            WHERE 
                h.datadate <= DATE '{date_str}'
                AND h.indfmt = 'INDL'
                AND h.datafmt = 'STD'
                AND h.consol = 'C'
                AND h.popsrc = 'D'
                AND h.curcdq = 'USD'
                AND h.updq = 3
                -- First adjust by years back
                AND (
                    -- If we need to go back one more year due to quarter adjustment
                    (cd.fqtr - {self.extra_quarters} <= 0 AND h.fyearq = cd.fyearq - {self.years_back} - 1)
                    OR
                    -- Standard year adjustment
                    (cd.fqtr - {self.extra_quarters} > 0 AND h.fyearq = cd.fyearq - {self.years_back})
                )
                -- Then adjust fiscal quarter based on extra quarters
                AND h.fqtr = CASE 
                    WHEN cd.fqtr - {self.extra_quarters} <= 0 
                    THEN cd.fqtr - {self.extra_quarters} + 4
                    ELSE cd.fqtr - {self.extra_quarters}
                END
            QUALIFY qrank = 1
        ),
        historical_quarter AS (
            SELECT *,
            -- Calculate quarterly OCF from YTD values
            CASE 
                WHEN fqtr = 1 THEN oancfy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_ocf IS NULL THEN oancfy  -- No previous quarter data
                ELSE oancfy - prev_quarter_ytd_ocf  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_ocf
            FROM historical_data
        ),
        combined_data AS (
            SELECT
                cq.gvkey,
                cq.datadate as current_date,
                cq.fyearq as current_fyear,
                cq.fqtr as current_fqtr,
                hq.datadate as historical_date,
                hq.fyearq as historical_fyear,
                hq.fqtr as historical_fqtr,
                DATEDIFF('day', hq.datadate, cq.datadate) as days_diff,
                -- Current period cash flow margin calculation
                CASE
                    WHEN cq.saleq IS NULL OR cq.saleq = 0 THEN NULL
                    WHEN cq.quarterly_ocf IS NULL THEN NULL
                    ELSE cq.quarterly_ocf / cq.saleq
                END as current_cash_flow_margin,
                -- Historical period cash flow margin calculation
                CASE
                    WHEN hq.saleq IS NULL OR hq.saleq = 0 THEN NULL
                    WHEN hq.quarterly_ocf IS NULL THEN NULL
                    ELSE hq.quarterly_ocf / hq.saleq
                END as historical_cash_flow_margin
            FROM current_quarter cq
            LEFT JOIN historical_quarter hq ON cq.gvkey = hq.gvkey
        )
        SELECT 
            -- Select a single PERMNO record per GVKEY to avoid duplicates
            t.permno, t.lpermno, t.lpermco, t.gvkey, t.iid,
            c.*
        FROM (
            -- For each GVKEY, select only ONE corresponding PERMNO record
            SELECT DISTINCT ON (gvkey) permno, lpermno, lpermco, gvkey, iid
            FROM temp_universe
        ) t
        LEFT JOIN combined_data c ON t.gvkey = c.gvkey
        """
        
        try:
            result_df = duck_conn.execute(query).fetchdf()
            
            # Make doubly sure we don't have duplicates
            if len(result_df) > 0 and 'permno' in result_df.columns:
                result_df = result_df.drop_duplicates(subset=['permno'])
            
            # Calculate change in cash flow margin (absolute difference)
            result_df[self.factor_name] = result_df.apply(
                lambda x: (x['current_cash_flow_margin'] - x['historical_cash_flow_margin'])
                if (pd.notnull(x['historical_cash_flow_margin']) and 
                    pd.notnull(x['current_cash_flow_margin']))
                else None,
                axis=1
            )
            
            self._print_summary(result_df, date_str)
            return result_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 
                            'current_date', self.factor_name]].rename(
                                columns={'current_date': 'date'})
        finally:
            duck_conn.unregister('temp_universe')    

class AssetGrowthCalculator(GrowthCalculator):
    """
    Calculator for Asset Growth with flexible lookback periods.
    
    Asset Growth = (Total Assets_t - Total Assets_t-1) / Total Assets_t-1
    
    This measures the percentage change in a company's total assets compared to a previous period,
    indicating expansion or contraction of the company's resource base.
    
    Assets growth is particularly important as it has been shown to have a negative
    relationship with future stock returns in many studies.
    
    Compustat fields:
    - Total Assets: atq
    """
    def __init__(self, lookback_quarters=4):
        super().__init__(
            factor_name=f"asset_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="Asset",
            use_pct_change=True  # Calculate as percentage change
        )
    
    def _get_financial_columns(self):
        return """
            cd.atq as current_assets,
            hd.atq as historical_assets
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        assets = row.get(f"{prefix}assets")
        
        if assets is None or pd.isna(assets) or assets <= 0:
            return None
        return assets
    
    def _print_summary_statistics(self, df, date_str):
        """Extended summary statistics specifically for asset growth"""
        super()._print_summary(df, date_str)
        
        # Additional asset growth specific analysis
        valid_growth = df.dropna(subset=[self.factor_name])
        
        if len(valid_growth) > 0:
            print("\nAsset Growth Distribution Details:")
            
            # Calculate additional percentiles for more granular view
            percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            print(df[self.factor_name].describe(percentiles=percentiles))
            
            # Count and percentage of firms with negative asset growth
            neg_growth = (valid_growth[self.factor_name] < 0).sum()
            neg_growth_pct = (neg_growth / len(valid_growth)) * 100
            print(f"\nFirms with negative asset growth: {neg_growth} ({neg_growth_pct:.2f}%)")
            
            # Count and percentage of firms with high asset growth (>20%)
            high_growth = (valid_growth[self.factor_name] > 0.2).sum()
            high_growth_pct = (high_growth / len(valid_growth)) * 100
            print(f"Firms with high asset growth (>20%): {high_growth} ({high_growth_pct:.2f}%)")


class ResidualFF3MomentumCalculator(FactorCalculator):
    """
    Calculator for residual momentum based on Fama-French 3-factor model.
    
    This implementation follows the methodology of residual momentum where:
    1. Factor loadings (betas) are estimated over a formation period
    2. These factor loadings are then used to calculate residual returns over a momentum period
    3. Residual momentum is calculated as the product of (1+residual_return)
    
    Parameters:
    -----------
    formation_days: Number of days to use for estimating factor loadings (e.g., 756 days ≈ 36 months)
    momentum_days: Number of days to use for calculating momentum (e.g., 252 days ≈ 12 months)
    skip_days: Number of days to skip between formation and momentum periods (default: 0)
    min_observations_formation: Minimum observations required in formation period
    min_observations_momentum: Minimum observations required in momentum period
    """
    def __init__(self, formation_days=1050, momentum_days=360, skip_days=30, 
                 min_observations_formation=600, min_observations_momentum=200):
        factor_name = f"res_mom_f{formation_days}d_m{momentum_days}d"
        if skip_days > 0:
            factor_name += f"_s{skip_days}d"
        
        super().__init__(factor_name)
        self.formation_days = formation_days
        self.momentum_days = momentum_days
        self.skip_days = skip_days
        self.min_observations_formation = min_observations_formation
        self.min_observations_momentum = min_observations_momentum
        self.factor_name = factor_name
    
    def calculate(self, date_str, universe_df, duck_conn):
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        
        # Calculate date ranges
        end_date = pd.to_datetime(date_str)
        momentum_start = end_date - pd.Timedelta(days=self.momentum_days)
        
        if self.skip_days > 0:
            formation_end = momentum_start - pd.Timedelta(days=self.skip_days)
        else:
            formation_end = momentum_start
            
        formation_start = formation_end - pd.Timedelta(days=self.formation_days)
        
        # Convert dates to strings for SQL
        end_date_str = end_date.strftime('%Y-%m-%d')
        momentum_start_str = momentum_start.strftime('%Y-%m-%d')
        formation_end_str = formation_end.strftime('%Y-%m-%d')
        formation_start_str = formation_start.strftime('%Y-%m-%d')
        
        # Register universe DataFrame
        duck_conn.register('temp_universe', universe_df)
        
        # 1. Fetch formation period data for estimating factor loadings
        formation_query = f"""
        SELECT 
            sr.permno,
            sr.dlycaldt as date,
            sr.DlyRet,
            ff.mktrf,
            ff.smb,
            ff.hml,
            ff.rf,
            (sr.DlyRet - ff.rf) as excess_return
        FROM stkdlysecuritydata sr
        INNER JOIN temp_universe u ON sr.permno = u.permno
        INNER JOIN read_parquet('Data_all/FF_data/ff_factors.parquet') ff ON sr.dlycaldt = ff.date
        WHERE sr.dlycaldt > DATE '{formation_start_str}'
          AND sr.dlycaldt <= DATE '{formation_end_str}'
          AND sr.DlyRet IS NOT NULL
        ORDER BY sr.permno, sr.dlycaldt
        """
        
        # 2. Fetch momentum period data for calculating residual returns
        momentum_query = f"""
        SELECT 
            sr.permno,
            sr.dlycaldt as date,
            sr.DlyRet,
            ff.mktrf,
            ff.smb,
            ff.hml,
            ff.rf,
            (sr.DlyRet - ff.rf) as excess_return
        FROM stkdlysecuritydata sr
        INNER JOIN temp_universe u ON sr.permno = u.permno
        INNER JOIN read_parquet('Data_all/FF_data/ff_factors.parquet') ff ON sr.dlycaldt = ff.date
        WHERE sr.dlycaldt > DATE '{momentum_start_str}'
          AND sr.dlycaldt <= DATE '{end_date_str}'
          AND sr.DlyRet IS NOT NULL
        ORDER BY sr.permno, sr.dlycaldt
        """
        
        # Execute queries
        formation_df = duck_conn.execute(formation_query).fetchdf()
        
        momentum_df = duck_conn.execute(momentum_query).fetchdf()
        
        # Estimate factor loadings and calculate residual momentum
        results = []
        
        # Get unique permnos
        permnos = universe_df['permno'].unique()
        
        for permno in permnos:
            try:
                # Get formation data for this stock
                formation_stock = formation_df[formation_df['permno'] == permno]
                
                # Only proceed if we have enough observations
                if len(formation_stock) < self.min_observations_formation:
                    continue
                
                # Estimate factor loadings using formation period
                y = formation_stock['excess_return']
                X = formation_stock[['mktrf', 'smb', 'hml']]
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                
                # Get momentum period data
                momentum_stock = momentum_df[momentum_df['permno'] == permno]
                
                # Only proceed if we have enough observations
                if len(momentum_stock) < self.min_observations_momentum:
                    continue
                
                # Calculate residual returns for momentum period
                X_momentum = momentum_stock[['mktrf', 'smb', 'hml']]
                X_momentum = sm.add_constant(X_momentum)
                
                # Expected returns based on factor loadings
                expected_returns = X_momentum.dot(model.params)
                
                # Residual returns
                residual_returns = momentum_stock['excess_return'] - expected_returns
                
                # Calculate residual momentum as product of (1+residual_return)
                res_momentum = np.prod(1 + residual_returns) - 1
                
                # Store results
                results.append({
                    'permno': permno,
                    'formation_obs': len(formation_stock),
                    'momentum_obs': len(momentum_stock),
                    'alpha': model.params['const'],
                    'beta_mkt': model.params['mktrf'],
                    'beta_smb': model.params['smb'],
                    'beta_hml': model.params['hml'],
                    self.factor_name: res_momentum
                })
                
            except Exception as e:
                print(f"Error processing permno {permno}: {str(e)}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Return results with all stocks in universe
        if len(results_df) > 0:
            duck_conn.register('res_momentum_results', results_df)
            
            final_query = f"""
            SELECT 
                u.permno,
                u.lpermno,
                u.lpermco,
                u.gvkey,
                u.iid,
                DATE '{date_str}' as date,
                r.{self.factor_name}
            FROM temp_universe u
            LEFT JOIN res_momentum_results r ON u.permno = r.permno
            """
            
            final_result = duck_conn.execute(final_query).fetchdf()
        else:
            # If no results, return just universe with missing factor values
            final_result = universe_df.copy()
            final_result['date'] = pd.to_datetime(date_str)
            final_result[self.factor_name] = np.nan
        
        # Clean up
        duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
        
        self._print_summary_statistics(results_df, date_str)
        
        return final_result
    
    def _print_summary_statistics(self, df, date_str):
        if df.empty:
            print(f"\nNo valid residual momentum results for {date_str}")
            return
            
        total_stocks = len(df)
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Stocks with valid residual momentum: {total_stocks}")
        
        print("\nResidual Momentum Distribution:")
        print(df[self.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nFactor Loadings from Formation Period:")
        for col in ['alpha', 'beta_mkt', 'beta_smb', 'beta_hml']:
            print(f"\n{col} statistics:")
            print(df[col].describe(percentiles=[.05, .25, .5, .75, .95]))
                
        print("\nObservation Counts:")
        print("Formation Period:")
        print(df['formation_obs'].describe(percentiles=[.05, .25, .5, .75, .95]))
        print("\nMomentum Period:")
        print(df['momentum_obs'].describe(percentiles=[.05, .25, .5, .75, .95]))    



class BetaCalculator(FactorCalculator):
    """
    Calculator for stock betas over a specified lookback period.
    
    This calculator computes market betas for each stock by regressing
    stock excess returns against market excess returns over a specified 
    lookback period.
    
    Parameters:
    -----------
    lookback_days: Number of trading days to use for beta estimation
    min_observations: Minimum number of observations required for regression
    include_additional_factors: Whether to include SMB and HML factors (FF3 model)
    """
    def __init__(self, lookback_days=252, min_observations=60, include_additional_factors=False):
        if include_additional_factors:
            factor_name = f"ff3_beta_{lookback_days}d"
        else:
            factor_name = f"beta_{lookback_days}d"
        
        super().__init__(factor_name)
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.include_additional_factors = include_additional_factors
        self.factor_name = factor_name
    
    def calculate(self, date_str, universe_df, duck_conn):
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        
        # Register universe DataFrame
        duck_conn.register('temp_universe', universe_df)
        
        # First, get all available trading dates up to the calculation date
        date_query = f"""
        SELECT DISTINCT dlycaldt
        FROM stkdlysecuritydata
        WHERE dlycaldt <= DATE '{date_str}'
        ORDER BY dlycaldt DESC
        """
        
        trading_dates = duck_conn.execute(date_query).fetchdf()
        trading_dates = trading_dates['dlycaldt'].tolist()
        
        # Determine the exact cutoff date for the lookback period
        if len(trading_dates) < self.lookback_days + 1:
            print(f"Not enough trading dates available before {date_str}")
            return universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']].copy()
        
        # Get the exact date for period start
        lookback_start_date = trading_dates[self.lookback_days]
        
        # Format date for query
        lookback_start_str = lookback_start_date.strftime('%Y-%m-%d') if isinstance(lookback_start_date, pd.Timestamp) else lookback_start_date
        
        # Fetch data for beta calculation
        data_query = f"""
        SELECT 
            sr.permno,
            sr.dlycaldt as date,
            sr.DlyRet,
            ff.mktrf,
            ff.smb,
            ff.hml,
            ff.rf,
            (sr.DlyRet - ff.rf) as excess_return
        FROM stkdlysecuritydata sr
        INNER JOIN temp_universe u ON sr.permno = u.permno
        INNER JOIN read_parquet('Data_all/FF_data/ff_factors.parquet') ff ON sr.dlycaldt = ff.date
        WHERE sr.dlycaldt >= DATE '{lookback_start_str}'
          AND sr.dlycaldt <= DATE '{date_str}'
          AND sr.DlyRet IS NOT NULL
        ORDER BY sr.permno, sr.dlycaldt
        """
        
        # Execute query
        data_df = duck_conn.execute(data_query).fetchdf()
        
        # Calculate betas for each stock
        results = []
        permnos = universe_df['permno'].unique()
        
        for permno in permnos:
            try:
                # Get data for this stock
                stock_data = data_df[data_df['permno'] == permno]
                
                # Only proceed if we have enough observations
                if len(stock_data) < self.min_observations:
                    continue
                
                # Set up regression
                y = stock_data['excess_return']
                
                if self.include_additional_factors:
                    # FF3 model (market, size, value)
                    X = stock_data[['mktrf', 'smb', 'hml']]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    
                    results.append({
                        'permno': permno,
                        'num_observations': len(stock_data),
                        'beta_market': model.params['mktrf'],
                        'beta_smb': model.params['smb'],
                        'beta_hml': model.params['hml'],
                        'alpha': model.params['const'],
                        'r_squared': model.rsquared,
                        self.factor_name: model.params['mktrf']  # Market beta is the primary factor
                    })
                else:
                    # CAPM model (market only)
                    X = stock_data[['mktrf']]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    
                    results.append({
                        'permno': permno,
                        'num_observations': len(stock_data),
                        'beta_market': model.params['mktrf'],
                        'alpha': model.params['const'],
                        'r_squared': model.rsquared,
                        self.factor_name: model.params['mktrf']
                    })
                
            except Exception as e:
                print(f"Error processing permno {permno}: {str(e)}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Return results with all stocks in universe
        if len(results_df) > 0:
            duck_conn.register('beta_results', results_df)
            
            if self.include_additional_factors:
                final_query = f"""
                SELECT 
                    u.permno,
                    u.lpermno,
                    u.lpermco,
                    u.gvkey,
                    u.iid,
                    DATE '{date_str}' as date,
                    r.beta_market,
                    r.beta_smb,
                    r.beta_hml,
                    r.alpha,
                    r.r_squared,
                    r.{self.factor_name}
                FROM temp_universe u
                LEFT JOIN beta_results r ON u.permno = r.permno
                """
            else:
                final_query = f"""
                SELECT 
                    u.permno,
                    u.lpermno,
                    u.lpermco,
                    u.gvkey,
                    u.iid,
                    DATE '{date_str}' as date,
                    r.beta_market,
                    r.alpha,
                    r.r_squared,
                    r.{self.factor_name}
                FROM temp_universe u
                LEFT JOIN beta_results r ON u.permno = r.permno
                """
            
            final_result = duck_conn.execute(final_query).fetchdf()
        else:
            # If no results, return just universe with missing factor values
            final_result = universe_df.copy()
            final_result['date'] = pd.to_datetime(date_str)
            final_result[self.factor_name] = np.nan
        
        # Clean up
        duck_conn.execute('DROP VIEW IF EXISTS temp_universe')
        
        # Print period information
        print(f"\nBeta calculation period for {date_str}:")
        print(f"Lookback period: {lookback_start_str} to {date_str} ({self.lookback_days} trading days)")
        
        self._print_summary_statistics(results_df, date_str)
        
        return final_result
    
    def _print_summary_statistics(self, df, date_str):
        if df.empty:
            print(f"\nNo valid beta results for {date_str}")
            return
            
        total_stocks = len(df)
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Stocks with valid beta estimates: {total_stocks}")
        
        print("\nMarket Beta Distribution:")
        print(df['beta_market'].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        if self.include_additional_factors:
            print("\nSMB Beta Distribution:")
            print(df['beta_smb'].describe(percentiles=[.05, .25, .5, .75, .95]))
            
            print("\nHML Beta Distribution:")
            print(df['beta_hml'].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nAlpha Distribution:")
        print(df['alpha'].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nR-squared Distribution:")
        print(df['r_squared'].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nNumber of observations:")
        print(df['num_observations'].describe(percentiles=[.05, .25, .5, .75, .95]))        


class CapExGrowthCalculator(GrowthCalculator):
    """Capital Expenditure growth with YTD conversion and flexible lookback periods"""
    def __init__(self, lookback_quarters=4, use_pct_change=False):
        super().__init__(
            factor_name=f"capex_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="CAPEX",
            use_pct_change=use_pct_change
        )

    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']])
        
        # For the historical period, we adjust by years_back and then adjust the fiscal quarter
        # based on the extra quarters
        query = f"""
        WITH current_year_data AS (
            SELECT 
                c.*,
                LAG(c.capxy) OVER (PARTITION BY c.gvkey ORDER BY c.datadate) as prev_quarter_ytd_capex,
                ROW_NUMBER() OVER (PARTITION BY c.gvkey ORDER BY c.datadate DESC) as qrank
            FROM wrds_csq_pit c
            INNER JOIN (
                SELECT DISTINCT gvkey FROM temp_universe
            ) u ON c.gvkey = u.gvkey
            WHERE 
                c.datadate <= DATE '{date_str}'
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
            QUALIFY qrank <= 5  -- Get current quarter + previous for YTD calc
        ),
        current_quarter AS (
            SELECT *,
            -- Calculate quarterly CAPEX from YTD values
            CASE 
                WHEN fqtr = 1 THEN capxy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_capex IS NULL THEN capxy  -- No previous quarter data
                ELSE capxy - prev_quarter_ytd_capex  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_capex
            FROM current_year_data
            WHERE qrank = 1
        ),
        historical_data AS (
            SELECT 
                h.*,
                LAG(h.capxy) OVER (PARTITION BY h.gvkey ORDER BY h.datadate) as prev_quarter_ytd_capex,
                ROW_NUMBER() OVER (PARTITION BY h.gvkey, h.fyearq, h.fqtr ORDER BY h.datadate DESC) as qrank
            FROM wrds_csq_pit h
            INNER JOIN current_quarter cd ON h.gvkey = cd.gvkey 
            WHERE 
                h.datadate <= DATE '{date_str}'
                AND h.indfmt = 'INDL'
                AND h.datafmt = 'STD'
                AND h.consol = 'C'
                AND h.popsrc = 'D'
                AND h.curcdq = 'USD'
                AND h.updq = 3
                -- First adjust by years back
                AND (
                    -- If we need to go back one more year due to quarter adjustment
                    (cd.fqtr - {self.extra_quarters} <= 0 AND h.fyearq = cd.fyearq - {self.years_back} - 1)
                    OR
                    -- Standard year adjustment
                    (cd.fqtr - {self.extra_quarters} > 0 AND h.fyearq = cd.fyearq - {self.years_back})
                )
                -- Then adjust fiscal quarter based on extra quarters
                AND h.fqtr = CASE 
                    WHEN cd.fqtr - {self.extra_quarters} <= 0 
                    THEN cd.fqtr - {self.extra_quarters} + 4
                    ELSE cd.fqtr - {self.extra_quarters}
                END
            QUALIFY qrank = 1
        ),
        historical_quarter AS (
            SELECT *,
            -- Calculate quarterly CAPEX from YTD values
            CASE 
                WHEN fqtr = 1 THEN capxy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_capex IS NULL THEN capxy  -- No previous quarter data
                ELSE capxy - prev_quarter_ytd_capex  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_capex
            FROM historical_data
        ),
        combined_data AS (
            SELECT
                cq.gvkey,
                cq.datadate as current_date,
                cq.fyearq as current_fyear,
                cq.fqtr as current_fqtr,
                hq.datadate as historical_date,
                hq.fyearq as historical_fyear,
                hq.fqtr as historical_fqtr,
                -- Current period CAPEX - raw values only, no normalization
                cq.quarterly_capex as current_capex,
                -- Historical period CAPEX - raw values only, no normalization
                hq.quarterly_capex as historical_capex,
                DATEDIFF('day', hq.datadate, cq.datadate) as days_diff
            FROM current_quarter cq
            LEFT JOIN historical_quarter hq ON cq.gvkey = hq.gvkey
        )
        SELECT 
            -- Select a single PERMNO record per GVKEY to avoid duplicates
            t.permno, t.lpermno, t.lpermco, t.gvkey, t.iid,
            c.*
        FROM (
            -- For each GVKEY, select only ONE corresponding PERMNO record
            SELECT DISTINCT ON (gvkey) permno, lpermno, lpermco, gvkey, iid
            FROM temp_universe
        ) t
        LEFT JOIN combined_data c ON t.gvkey = c.gvkey
        """
        
        try:
            result_df = duck_conn.execute(query).fetchdf()
            
            # Make doubly sure we don't have duplicates
            if len(result_df) > 0 and 'permno' in result_df.columns:
                result_df = result_df.drop_duplicates(subset=['permno'])
            
            # Calculate growth rate based on choice between percentage or difference
            # Default version uses raw CapEx values
            if self.use_pct_change:
                result_df[self.factor_name] = result_df.apply(
                    lambda x: (x['current_capex'] - x['historical_capex'])/abs(x['historical_capex'])
                    if (pd.notnull(x['historical_capex']) and 
                        pd.notnull(x['current_capex']) and 
                        abs(x['historical_capex']) > 1e-6)
                    else None,
                    axis=1
                )
            else:
                result_df[self.factor_name] = result_df.apply(
                    lambda x: (x['current_capex'] - x['historical_capex'])
                    if (pd.notnull(x['historical_capex']) and 
                        pd.notnull(x['current_capex']))
                    else None,
                    axis=1
                )
            
            self._print_summary(result_df, date_str)
            return result_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 
                            'current_date', self.factor_name]].rename(
                                columns={'current_date': 'date'})
        finally:
            duck_conn.unregister('temp_universe')

# You could also create a normalized version that calculates CapEx/Assets growth
class CapExToAssetsGrowthCalculator(GrowthCalculator):
    """Capital Expenditure to Assets ratio growth with YTD conversion and flexible lookback periods"""
    def __init__(self, lookback_quarters=4, use_pct_change=False):
        super().__init__(
            factor_name=f"capex_to_assets_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="CAPEX/ASSETS",
            use_pct_change=use_pct_change
        )

    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']])
        
        # Same query structure as CapExGrowthCalculator but using the normalized metrics for the factor calculation
        query = f"""
        WITH current_year_data AS (
            SELECT 
                c.*,
                LAG(c.capxy) OVER (PARTITION BY c.gvkey ORDER BY c.datadate) as prev_quarter_ytd_capex,
                ROW_NUMBER() OVER (PARTITION BY c.gvkey ORDER BY c.datadate DESC) as qrank
            FROM wrds_csq_pit c
            INNER JOIN (
                SELECT DISTINCT gvkey FROM temp_universe
            ) u ON c.gvkey = u.gvkey
            WHERE 
                c.datadate <= DATE '{date_str}'
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
            QUALIFY qrank <= 5  -- Get current quarter + previous for YTD calc
        ),
        current_quarter AS (
            SELECT *,
            -- Calculate quarterly CAPEX from YTD values
            CASE 
                WHEN fqtr = 1 THEN capxy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_capex IS NULL THEN capxy  -- No previous quarter data
                ELSE capxy - prev_quarter_ytd_capex  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_capex
            FROM current_year_data
            WHERE qrank = 1
        ),
        historical_data AS (
            SELECT 
                h.*,
                LAG(h.capxy) OVER (PARTITION BY h.gvkey ORDER BY h.datadate) as prev_quarter_ytd_capex,
                ROW_NUMBER() OVER (PARTITION BY h.gvkey, h.fyearq, h.fqtr ORDER BY h.datadate DESC) as qrank
            FROM wrds_csq_pit h
            INNER JOIN current_quarter cd ON h.gvkey = cd.gvkey 
            WHERE 
                h.datadate <= DATE '{date_str}'
                AND h.indfmt = 'INDL'
                AND h.datafmt = 'STD'
                AND h.consol = 'C'
                AND h.popsrc = 'D'
                AND h.curcdq = 'USD'
                AND h.updq = 3
                -- First adjust by years back
                AND (
                    -- If we need to go back one more year due to quarter adjustment
                    (cd.fqtr - {self.extra_quarters} <= 0 AND h.fyearq = cd.fyearq - {self.years_back} - 1)
                    OR
                    -- Standard year adjustment
                    (cd.fqtr - {self.extra_quarters} > 0 AND h.fyearq = cd.fyearq - {self.years_back})
                )
                -- Then adjust fiscal quarter based on extra quarters
                AND h.fqtr = CASE 
                    WHEN cd.fqtr - {self.extra_quarters} <= 0 
                    THEN cd.fqtr - {self.extra_quarters} + 4
                    ELSE cd.fqtr - {self.extra_quarters}
                END
            QUALIFY qrank = 1
        ),
        historical_quarter AS (
            SELECT *,
            -- Calculate quarterly CAPEX from YTD values
            CASE 
                WHEN fqtr = 1 THEN capxy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_capex IS NULL THEN capxy  -- No previous quarter data
                ELSE capxy - prev_quarter_ytd_capex  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_capex
            FROM historical_data
        ),
        combined_data AS (
            SELECT
                cq.gvkey,
                cq.datadate as current_date,
                cq.fyearq as current_fyear,
                cq.fqtr as current_fqtr,
                hq.datadate as historical_date,
                hq.fyearq as historical_fyear,
                hq.fqtr as historical_fqtr,
                -- Current period CAPEX normalized by assets
                CASE
                    WHEN cq.atq IS NULL OR cq.atq = 0 THEN NULL
                    WHEN cq.quarterly_capex IS NULL THEN NULL
                    ELSE cq.quarterly_capex / cq.atq
                END as current_capex_to_assets,
                -- Historical period CAPEX normalized by assets
                CASE
                    WHEN hq.atq IS NULL OR hq.atq = 0 THEN NULL
                    WHEN hq.quarterly_capex IS NULL THEN NULL
                    ELSE hq.quarterly_capex / hq.atq
                END as historical_capex_to_assets,
                DATEDIFF('day', hq.datadate, cq.datadate) as days_diff
            FROM current_quarter cq
            LEFT JOIN historical_quarter hq ON cq.gvkey = hq.gvkey
        )
        SELECT 
            -- Select a single PERMNO record per GVKEY to avoid duplicates
            t.permno, t.lpermno, t.lpermco, t.gvkey, t.iid,
            c.*
        FROM (
            -- For each GVKEY, select only ONE corresponding PERMNO record
            SELECT DISTINCT ON (gvkey) permno, lpermno, lpermco, gvkey, iid
            FROM temp_universe
        ) t
        LEFT JOIN combined_data c ON t.gvkey = c.gvkey
        """
        
        try:
            result_df = duck_conn.execute(query).fetchdf()
            
            # Make doubly sure we don't have duplicates
            if len(result_df) > 0 and 'permno' in result_df.columns:
                result_df = result_df.drop_duplicates(subset=['permno'])
            
            # Calculate growth rate based on the normalized metrics
            if self.use_pct_change:
                result_df[self.factor_name] = result_df.apply(
                    lambda x: (x['current_capex_to_assets'] - x['historical_capex_to_assets'])/abs(x['historical_capex_to_assets'])
                    if (pd.notnull(x['historical_capex_to_assets']) and 
                        pd.notnull(x['current_capex_to_assets']) and 
                        abs(x['historical_capex_to_assets']) > 1e-6)
                    else None,
                    axis=1
                )
            else:
                result_df[self.factor_name] = result_df.apply(
                    lambda x: (x['current_capex_to_assets'] - x['historical_capex_to_assets'])
                    if (pd.notnull(x['historical_capex_to_assets']) and 
                        pd.notnull(x['current_capex_to_assets']))
                    else None,
                    axis=1
                )
            
            self._print_summary(result_df, date_str)
            return result_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 
                            'current_date', self.factor_name]].rename(
                                columns={'current_date': 'date'})
        finally:
            duck_conn.unregister('temp_universe')

class LogMarketCapCalculator(FactorCalculator):
    """
    Simple factor calculator that computes the natural logarithm of market capitalization.
    Using DlyCap field from CRSP daily stock data which is already in millions.
    """
    def __init__(self):
        super().__init__("log_market_cap")
        
    def calculate(self, date_str, universe_df, duck_conn):
        duck_conn.register('temp_universe', universe_df)
        
        query = """
        SELECT 
            u.permno,
            u.lpermno,
            u.lpermco,
            u.gvkey,
            u.iid,
            DATE '{date}' as date,
            s.DlyCap as market_cap
        FROM temp_universe u
        LEFT JOIN stkdlysecuritydata s 
        ON s.permno = u.permno
           AND s.dlycaldt = DATE '{date}'
        WHERE s.DlyCap IS NOT NULL
          AND s.DlyCap > 0
        """.format(date=date_str)
        
        try:
            result_df = duck_conn.execute(query).fetchdf()
            
            # Calculate log market cap
            result_df[self.factor_name] = np.log(result_df['market_cap'])
            
            # Print summary statistics
            self._print_summary(result_df, date_str)
            
            return result_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 
                            'date', self.factor_name]]
        finally:
            duck_conn.unregister('temp_universe')
    
    def _print_summary(self, df, date_str):
        """Print summary statistics for log market cap factor"""
        print(f"\n{'-'*40}")
        print(f"LOG MARKET CAP Summary ({date_str})")
        
        total_stocks = len(df)
        stocks_with_log_mcap = df[self.factor_name].notna().sum()
        
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with valid log market cap: {stocks_with_log_mcap}")
        
        if total_stocks > 0:
            print(f"Coverage ratio: {(stocks_with_log_mcap/total_stocks)*100:.2f}%")
        
        print(f"\nLog Market Cap distribution:")
        print(df[self.factor_name].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
        
        # Print market cap stats in millions for reference
        if 'market_cap' in df.columns:
            print(f"\nMarket Cap distribution (in millions):")
            print(df['market_cap'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))





class MonthlyResidualFF3MomentumCalculator(FactorCalculator):
    """
    Calculator for residual momentum based on Fama-French 3-factor model using monthly data.
    
    This implementation follows the methodology of residual momentum where:
    1. Factor loadings (betas) are estimated over a formation period using monthly returns
    2. These factor loadings are then used to calculate residual returns over a momentum period
    3. Residual momentum is calculated as the product of (1+residual_return)
    
    Parameters:
    -----------
    formation_months: Number of months to use for estimating factor loadings (e.g., 36 months)
    momentum_months: Number of months to use for calculating momentum (e.g., 11 months)
    skip_months: Number of months to skip between formation and momentum periods (default: 1)
    min_observations_formation: Minimum monthly observations required in formation period
    min_observations_momentum: Minimum monthly observations required in momentum period
    """
    def __init__(self, formation_months=36, momentum_months=11, skip_months=1, 
                 min_observations_formation=24, min_observations_momentum=8):
        factor_name = f"monthly_res_mom_f{formation_months}m_m{momentum_months}m"
        if skip_months > 0:
            factor_name += f"_s{skip_months}m"
        
        super().__init__(factor_name)
        self.formation_months = formation_months
        self.momentum_months = momentum_months
        self.skip_months = skip_months
        self.min_observations_formation = min_observations_formation
        self.min_observations_momentum = min_observations_momentum
        self.factor_name = factor_name
    
    def calculate(self, date_str, universe_df, duck_conn):
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        
        # Calculate date ranges
        end_date = pd.to_datetime(date_str)
        
        # Calculate the first day of current month
        current_month_start = pd.Timestamp(year=end_date.year, month=end_date.month, day=1)
        
        # Calculate the first day of the momentum period by subtracting momentum_months from current month
        momentum_start = current_month_start - pd.DateOffset(months=self.momentum_months)
        
        # Calculate the end of formation period (with skip if any)
        if self.skip_months > 0:
            formation_end = momentum_start - pd.DateOffset(months=self.skip_months)
        else:
            formation_end = momentum_start
        
        # Calculate the start of formation period
        formation_start = formation_end - pd.DateOffset(months=self.formation_months)
        
        # Convert dates to strings for SQL
        end_date_str = end_date.strftime('%Y-%m-%d')
        momentum_start_str = momentum_start.strftime('%Y-%m-%d')
        formation_end_str = formation_end.strftime('%Y-%m-%d')
        formation_start_str = formation_start.strftime('%Y-%m-%d')
        
        # Register universe DataFrame
        duck_conn.register('temp_universe', universe_df)
        
        # 1. Fetch daily data for formation period and convert to monthly returns
        # Using PRODUCT instead of AVG to calculate factor returns as well
        formation_query = f"""
        WITH daily_returns AS (
            SELECT 
                sr.permno,
                sr.dlycaldt as date,
                sr.DlyRet,
                ff.mktrf,
                ff.smb,
                ff.hml,
                ff.rf,
                (sr.DlyRet - ff.rf) as excess_return,
                (1 + rf) as rf_return,
                (1 + mktrf) as mktrf_return,
                (1 + smb) as smb_return,
                (1 + hml) as hml_return,
                DATE_TRUNC('month', sr.dlycaldt) as month_start
            FROM stkdlysecuritydata sr
            INNER JOIN temp_universe u ON sr.permno = u.permno
            INNER JOIN read_parquet('Data_all/FF_data/ff_factors.parquet') ff ON sr.dlycaldt = ff.date
            WHERE sr.dlycaldt >= DATE '{formation_start_str}'
              AND sr.dlycaldt < DATE '{formation_end_str}'
              AND sr.DlyRet IS NOT NULL
        ),
        monthly_returns AS (
            SELECT 
                permno,
                month_start,
                (PRODUCT(1 + DlyRet) - 1) as monthly_ret,
                (PRODUCT(1 + excess_return) - 1) as monthly_excess_ret,
                (PRODUCT(mktrf_return) - 1) as monthly_mktrf,  -- Cumulative return
                (PRODUCT(smb_return) - 1) as monthly_smb,      -- Cumulative return
                (PRODUCT(hml_return) - 1) as monthly_hml,      -- Cumulative return
                (PRODUCT(rf_return) - 1) as monthly_rf,        -- Cumulative return
                COUNT(*) as days_in_month
            FROM daily_returns
            GROUP BY permno, month_start
            HAVING COUNT(*) >= 15  -- Require at least 15 trading days in a month
        )
        SELECT * FROM monthly_returns
        ORDER BY permno, month_start
        """
        
        # 2. Fetch daily data for momentum period and convert to monthly returns
        # Using PRODUCT instead of AVG for factor returns here as well
        momentum_query = f"""
        WITH daily_returns AS (
            SELECT 
                sr.permno,
                sr.dlycaldt as date,
                sr.DlyRet,
                ff.mktrf,
                ff.smb,
                ff.hml,
                ff.rf,
                (sr.DlyRet - ff.rf) as excess_return,
                (1 + mktrf) as mktrf_return,
                (1 + smb) as smb_return,
                (1 + hml) as hml_return,
                (1 + ff.rf) as rf_return,
                DATE_TRUNC('month', sr.dlycaldt) as month_start
            FROM stkdlysecuritydata sr
            INNER JOIN temp_universe u ON sr.permno = u.permno
            INNER JOIN read_parquet('Data_all/FF_data/ff_factors.parquet') ff ON sr.dlycaldt = ff.date
            WHERE sr.dlycaldt >= DATE '{momentum_start_str}'
              AND sr.dlycaldt <= DATE '{end_date_str}'
              AND sr.DlyRet IS NOT NULL
        ),
        monthly_returns AS (
            SELECT 
                permno,
                month_start,
                (PRODUCT(1 + DlyRet) - 1) as monthly_ret,
                (PRODUCT(1 + excess_return) - 1) as monthly_excess_ret,
                (PRODUCT(mktrf_return) - 1) as monthly_mktrf,  -- Cumulative return
                (PRODUCT(smb_return) - 1) as monthly_smb,      -- Cumulative return
                (PRODUCT(hml_return) - 1) as monthly_hml,      -- Cumulative return
                AVG(rf) as monthly_rf,                         -- Keep average for risk-free rate
                COUNT(*) as days_in_month
            FROM daily_returns
            GROUP BY permno, month_start
            HAVING COUNT(*) >= 15  -- Require at least 15 trading days in a month
        )
        SELECT * FROM monthly_returns
        ORDER BY permno, month_start
        """
        
        # Execute queries
        formation_df = duck_conn.execute(formation_query).fetchdf()
        momentum_df = duck_conn.execute(momentum_query).fetchdf()
        
        # Estimate factor loadings and calculate residual momentum
        results = []
        
        # Get unique permnos
        permnos = universe_df['permno'].unique()
        
        for permno in permnos:
            try:
                # Get formation data for this stock
                formation_stock = formation_df[formation_df['permno'] == permno]
                
                # Only proceed if we have enough monthly observations
                if len(formation_stock) < self.min_observations_formation:
                    continue
                
                # Estimate factor loadings using formation period
                y = formation_stock['monthly_excess_ret']
                X = formation_stock[['monthly_mktrf', 'monthly_smb', 'monthly_hml']]
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                
                # Get momentum period data
                momentum_stock = momentum_df[momentum_df['permno'] == permno]
                
                # Only proceed if we have enough monthly observations
                if len(momentum_stock) < self.min_observations_momentum:
                    continue
                
                # Calculate residual returns for momentum period
                X_momentum = momentum_stock[['monthly_mktrf', 'monthly_smb', 'monthly_hml']]
                X_momentum = sm.add_constant(X_momentum)
                
                # Expected returns based on factor loadings
                expected_returns = X_momentum.dot(model.params)
                
                # Residual returns
                residual_returns = momentum_stock['monthly_excess_ret'] - expected_returns
                
                # Calculate residual momentum as product of (1+residual_return)
                res_momentum = np.prod(1 + residual_returns) - 1
                
                # Store results
                results.append({
                    'permno': permno,
                    'formation_obs': len(formation_stock),
                    'momentum_obs': len(momentum_stock),
                    'alpha': model.params['const'],
                    'beta_mkt': model.params['monthly_mktrf'],
                    'beta_smb': model.params['monthly_smb'],
                    'beta_hml': model.params['monthly_hml'],
                    self.factor_name: res_momentum
                })
                
            except Exception as e:
                print(f"Error processing permno {permno}: {str(e)}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Return results with all stocks in universe
        if len(results_df) > 0:
            duck_conn.register('res_momentum_results', results_df)
            
            final_query = f"""
            SELECT 
                u.permno,
                u.lpermno,
                u.lpermco,
                u.gvkey,
                u.iid,
                DATE '{date_str}' as date,
                r.{self.factor_name}
            FROM temp_universe u
            LEFT JOIN res_momentum_results r ON u.permno = r.permno
            """
            
            final_result = duck_conn.execute(final_query).fetchdf()
        else:
            # If no results, return just universe with missing factor values
            final_result = universe_df.copy()
            final_result['date'] = pd.to_datetime(date_str)
            final_result[self.factor_name] = np.nan
        
        # Clean up
        duck_conn.unregister('temp_universe')
        if len(results_df) > 0:
            duck_conn.unregister('res_momentum_results')
        
        self._print_summary_statistics(results_df, date_str)
        
        return final_result
    
    def _print_summary_statistics(self, df, date_str):
        if df.empty:
            print(f"\nNo valid monthly residual momentum results for {date_str}")
            return
            
        total_stocks = len(df)
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Stocks with valid monthly residual momentum: {total_stocks}")
        
        print("\nMonthly Residual Momentum Distribution:")
        print(df[self.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nFactor Loadings from Formation Period:")
        for col in ['alpha', 'beta_mkt', 'beta_smb', 'beta_hml']:
            print(f"\n{col} statistics:")
            print(df[col].describe(percentiles=[.05, .25, .5, .75, .95]))
                
        print("\nObservation Counts:")
        print("Formation Period (months):")
        print(df['formation_obs'].describe(percentiles=[.05, .25, .5, .75, .95]))
        print("\nMomentum Period (months):")
        print(df['momentum_obs'].describe(percentiles=[.05, .25, .5, .75, .95]))


class MAX5Calculator(FactorCalculator):
    """
    Calculator for the mean of top 5 daily returns in the past 21 trading days.
    
    Parameters:
    -----------
    lookback_days: Number of trading days to consider (default: 21 days ≈ 1 month)
    top_n: Number of top daily returns to average (default: 5)
    """
    def __init__(self, lookback_days=21, top_n=5):
        factor_name = f"MAX{top_n}_{lookback_days}d"
        
        super().__init__(factor_name)
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.factor_name = factor_name
    
    def calculate(self, date_str, universe_df, duck_conn):
        import pandas as pd
        import numpy as np
        
        # Register universe DataFrame
        duck_conn.register('temp_universe', universe_df)
        
        # First, get all available trading dates up to the calculation date
        date_query = f"""
        SELECT DISTINCT dlycaldt
        FROM stkdlysecuritydata
        WHERE dlycaldt <= DATE '{date_str}'
        ORDER BY dlycaldt DESC
        """
        
        trading_dates = duck_conn.execute(date_query).fetchdf()
        trading_dates = trading_dates['dlycaldt'].tolist()
        
        # Determine the exact cutoff date for the lookback period
        if len(trading_dates) < self.lookback_days + 1:
            print(f"Not enough trading dates available before {date_str}")
            return universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']].copy()
        
        # Get the exact date for period start
        lookback_start_date = trading_dates[self.lookback_days]
        
        # Format date for query
        lookback_start_str = lookback_start_date.strftime('%Y-%m-%d') if isinstance(lookback_start_date, pd.Timestamp) else lookback_start_date
        
        # Fetch data for MAX5 calculation
        data_query = f"""
        SELECT 
            sr.permno,
            sr.dlycaldt as date,
            sr.DlyRet
        FROM stkdlysecuritydata sr
        INNER JOIN temp_universe u ON sr.permno = u.permno
        WHERE sr.dlycaldt >= DATE '{lookback_start_str}'
          AND sr.dlycaldt <= DATE '{date_str}'
          AND sr.DlyRet IS NOT NULL
        ORDER BY sr.permno, sr.dlycaldt
        """
        
        # Execute query
        data_df = duck_conn.execute(data_query).fetchdf()
        
        # Calculate MAX5 for each stock
        results = []
        permnos = universe_df['permno'].unique()
        
        for permno in permnos:
            try:
                # Get data for this stock
                stock_data = data_df[data_df['permno'] == permno]
                
                # Only proceed if we have data
                if len(stock_data) == 0:
                    continue
                
                # Sort returns in descending order and take top N
                top_returns = stock_data.sort_values('dlyret', ascending=False)['dlyret'].head(self.top_n)
                
                # If we have less than top_n returns, use whatever we have
                if len(top_returns) < self.top_n:
                    if len(top_returns) == 0:
                        continue
                    max_n = top_returns.mean()
                else:
                    max_n = top_returns.mean()
                
                results.append({
                    'permno': permno,
                    'num_observations': len(stock_data),
                    'num_top_returns': len(top_returns),
                    self.factor_name: max_n
                })
                
            except Exception as e:
                print(f"Error processing permno {permno} for MAX5: {str(e)}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Return results with all stocks in universe
        if len(results_df) > 0:
            duck_conn.register('max5_results', results_df)
            
            final_query = f"""
            SELECT 
                u.permno,
                u.lpermno,
                u.lpermco,
                u.gvkey,
                u.iid,
                DATE '{date_str}' as date,
                r.{self.factor_name}
            FROM temp_universe u
            LEFT JOIN max5_results r ON u.permno = r.permno
            """
            
            final_result = duck_conn.execute(final_query).fetchdf()
        else:
            # If no results, return just universe with missing factor values
            final_result = universe_df.copy()
            final_result['date'] = pd.to_datetime(date_str)
            final_result[self.factor_name] = np.nan
        
        # Clean up
        duck_conn.execute('DROP VIEW IF EXISTS max5_results')
        
        # Print period information
        print(f"\nMAX{self.top_n} calculation period for {date_str}:")
        print(f"Lookback period: {lookback_start_str} to {date_str} ({self.lookback_days} trading days)")
        
        self._print_summary_statistics(results_df, date_str)
        
        return final_result
    
    def _print_summary_statistics(self, df, date_str):
        if df.empty:
            print(f"\nNo valid MAX{self.top_n} results for {date_str}")
            return
            
        total_stocks = len(df)
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Stocks with valid MAX{self.top_n} estimates: {total_stocks}")
        
        print(f"\nMAX{self.top_n} Distribution:")
        print(df[self.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nNumber of observations:")
        print(df['num_observations'].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nNumber of top returns used:")
        print(df['num_top_returns'].describe(percentiles=[.05, .25, .5, .75, .95]))


class VOLCalculator(FactorCalculator):
    """
    Calculator for the standard deviation of excess returns over the past year.
    
    Parameters:
    -----------
    lookback_days: Number of trading days to consider (default: 252 days ≈ 1 year)
    min_observations: Minimum number of observations required for calculation
    """
    def __init__(self, lookback_days=252, min_observations=126):
        factor_name = f"VOL_{lookback_days}d"
        
        super().__init__(factor_name)
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.factor_name = factor_name
    
    def calculate(self, date_str, universe_df, duck_conn):
        import pandas as pd
        import numpy as np
        
        # Register universe DataFrame
        duck_conn.register('temp_universe', universe_df)
        
        # First, get all available trading dates up to the calculation date
        date_query = f"""
        SELECT DISTINCT dlycaldt
        FROM stkdlysecuritydata
        WHERE dlycaldt <= DATE '{date_str}'
        ORDER BY dlycaldt DESC
        """
        
        trading_dates = duck_conn.execute(date_query).fetchdf()
        trading_dates = trading_dates['dlycaldt'].tolist()
        
        # Determine the exact cutoff date for the lookback period
        if len(trading_dates) < self.lookback_days + 1:
            print(f"Not enough trading dates available before {date_str}")
            return universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']].copy()
        
        # Get the exact date for period start
        lookback_start_date = trading_dates[self.lookback_days]
        
        # Format date for query
        lookback_start_str = lookback_start_date.strftime('%Y-%m-%d') if isinstance(lookback_start_date, pd.Timestamp) else lookback_start_date
        
        # Fetch data for VOL calculation
        data_query = f"""
        SELECT 
            sr.permno,
            sr.dlycaldt as date,
            sr.DlyRet,
            ff.rf,
            (sr.DlyRet - ff.rf) as excess_return
        FROM stkdlysecuritydata sr
        INNER JOIN temp_universe u ON sr.permno = u.permno
        INNER JOIN read_parquet('Data_all/FF_data/ff_factors.parquet') ff ON sr.dlycaldt = ff.date
        WHERE sr.dlycaldt >= DATE '{lookback_start_str}'
          AND sr.dlycaldt <= DATE '{date_str}'
          AND sr.DlyRet IS NOT NULL
        ORDER BY sr.permno, sr.dlycaldt
        """
        
        # Execute query
        data_df = duck_conn.execute(data_query).fetchdf()
        
        # Calculate VOL for each stock
        results = []
        permnos = universe_df['permno'].unique()
        
        for permno in permnos:
            try:
                # Get data for this stock
                stock_data = data_df[data_df['permno'] == permno]
                
                # Only proceed if we have enough observations
                if len(stock_data) < self.min_observations:
                    continue
                
                # Calculate standard deviation of excess returns
                vol = stock_data['excess_return'].std()
                
                results.append({
                    'permno': permno,
                    'num_observations': len(stock_data),
                    self.factor_name: vol
                })
                
            except Exception as e:
                print(f"Error processing permno {permno} for VOL: {str(e)}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Return results with all stocks in universe
        if len(results_df) > 0:
            duck_conn.register('vol_results', results_df)
            
            final_query = f"""
            SELECT 
                u.permno,
                u.lpermno,
                u.lpermco,
                u.gvkey,
                u.iid,
                DATE '{date_str}' as date,
                r.{self.factor_name}
            FROM temp_universe u
            LEFT JOIN vol_results r ON u.permno = r.permno
            """
            
            final_result = duck_conn.execute(final_query).fetchdf()
        else:
            # If no results, return just universe with missing factor values
            final_result = universe_df.copy()
            final_result['date'] = pd.to_datetime(date_str)
            final_result[self.factor_name] = np.nan
        
        # Clean up
        duck_conn.execute('DROP VIEW IF EXISTS vol_results')
        
        # Print period information
        print(f"\nVOL calculation period for {date_str}:")
        print(f"Lookback period: {lookback_start_str} to {date_str} ({self.lookback_days} trading days)")
        
        self._print_summary_statistics(results_df, date_str)
        
        return final_result
    
    def _print_summary_statistics(self, df, date_str):
        if df.empty:
            print(f"\nNo valid VOL results for {date_str}")
            return
            
        total_stocks = len(df)
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Stocks with valid VOL estimates: {total_stocks}")
        
        print("\nVOL Distribution:")
        print(df[self.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nNumber of observations:")
        print(df['num_observations'].describe(percentiles=[.05, .25, .5, .75, .95]))


class MAX5VOLRatioCalculator(FactorCalculator):
    """
    Calculator for the ratio of MAX5 to VOL (MAX5/VOL).
    
    This calculator depends on both MAX5 and VOL calculators.
    
    Parameters:
    -----------
    max5_days: Number of trading days for MAX5 calculation (default: 21 days ≈ 1 month)
    vol_days: Number of trading days for VOL calculation (default: 252 days ≈ 1 year)
    top_n: Number of top daily returns to average for MAX5 (default: 5)
    min_vol_observations: Minimum number of observations required for VOL calculation
    """
    def __init__(self, max5_days=21, vol_days=252, top_n=5, min_vol_observations=126):
        factor_name = f"MAX{top_n}VOL_RATIO"
        
        super().__init__(factor_name)
        self.max5_days = max5_days
        self.vol_days = vol_days
        self.top_n = top_n
        self.min_vol_observations = min_vol_observations
        self.factor_name = factor_name
        
        # Create the component calculators
        self.max5_calculator = MAX5Calculator(lookback_days=max5_days, top_n=top_n)
        self.vol_calculator = VOLCalculator(lookback_days=vol_days, min_observations=min_vol_observations)
    
    def calculate(self, date_str, universe_df, duck_conn):
        import pandas as pd
        import numpy as np
        
        # Calculate MAX5
        max5_results = self.max5_calculator.calculate(date_str, universe_df, duck_conn)
        
        # Calculate VOL
        vol_results = self.vol_calculator.calculate(date_str, universe_df, duck_conn)
        
        # Register both results for joining
        duck_conn.register('max5_results', max5_results)
        duck_conn.register('vol_results', vol_results)
        
        # Join the results and calculate the ratio
        ratio_query = f"""
        SELECT 
            m.permno,
            m.lpermno,
            m.lpermco,
            m.gvkey,
            m.iid,
            m.date,
            m.{self.max5_calculator.factor_name},
            v.{self.vol_calculator.factor_name},
            CASE 
                WHEN v.{self.vol_calculator.factor_name} > 0 
                THEN m.{self.max5_calculator.factor_name} / v.{self.vol_calculator.factor_name}
                ELSE NULL
            END AS {self.factor_name}
        FROM max5_results m
        LEFT JOIN vol_results v ON m.permno = v.permno
        """
        
        # Execute the query
        final_result = duck_conn.execute(ratio_query).fetchdf()
        
        # Clean up
        duck_conn.execute('DROP VIEW IF EXISTS max5_results')
        duck_conn.execute('DROP VIEW IF EXISTS vol_results')
        
        # Extract just the ratio results for summary statistics
        ratio_results = final_result[[
            'permno', 
            self.max5_calculator.factor_name, 
            self.vol_calculator.factor_name, 
            self.factor_name
        ]].dropna(subset=[self.factor_name])
        
        self._print_summary_statistics(ratio_results, date_str)
        
        return final_result
    
    def _print_summary_statistics(self, df, date_str):
        if df.empty:
            print(f"\nNo valid {self.factor_name} results for {date_str}")
            return
            
        total_stocks = len(df)
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Stocks with valid ratio estimates: {total_stocks}")
        
        print(f"\n{self.factor_name} Distribution:")
        print(df[self.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        # Also print MAX5 and VOL stats for these stocks
        print(f"\n{self.max5_calculator.factor_name} Distribution (for valid ratio stocks):")
        print(df[self.max5_calculator.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print(f"\n{self.vol_calculator.factor_name} Distribution (for valid ratio stocks):")
        print(df[self.vol_calculator.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))

class InventoryGrowthCalculator(GrowthCalculator):
    """Inventory growth with flexible lookback periods
    
    According to the Compustat definitions:
    - Inventories (INVT): Raw inventory value from Compustat
    """
    def __init__(self, lookback_quarters=4, use_pct_change=True):
        super().__init__(
            factor_name=f"inv_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="INV",
            use_pct_change=use_pct_change
        )

    def _get_financial_columns(self):
        return """
            cd.invtq as current_inventory,
            hd.invtq as historical_inventory
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        inventory = row.get(f"{prefix}inventory")
        
        if inventory is None:
            return None
        return inventory


class NetNonCurrentOperatingAssetsOverAssetsGrowthCalculator(GrowthCalculator):
    """Net Non-Current Operating Assets (NCOA - NCOL) over Assets growth with flexible lookback periods
    
    According to the Compustat definitions provided:
    - Non-Current Operating Assets (NCOA) = AT - CA - IVAO
    - Non-Current Operating Liabilities (NCOL) = LT - CL - DLTT
    - Net Non-Current Operating Assets = NCOA - NCOL
    - Ratio = (NCOA - NCOL) / AT
    """
    def __init__(self, lookback_quarters=4, use_pct_change=False):
        super().__init__(
            factor_name=f"net_ncoa_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="NET_NCOA",
            use_pct_change=use_pct_change
        )

    def _get_financial_columns(self):
        return """
            cd.atq as current_assets,
            cd.actq as current_current_assets,
            cd.ivaoq as current_investment_advances_other,
            cd.ltq as current_total_liabilities,
            cd.lctq as current_current_liabilities,
            cd.dlttq as current_long_term_debt,
            hd.atq as historical_assets,
            hd.actq as historical_current_assets, 
            hd.ivaoq as historical_investment_advances_other,
            hd.ltq as historical_total_liabilities,
            hd.lctq as historical_current_liabilities,
            hd.dlttq as historical_long_term_debt
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        total_assets = row.get(f"{prefix}assets")
        current_assets = row.get(f"{prefix}current_assets", 0) or 0
        investment_advances = row.get(f"{prefix}investment_advances_other", 0) or 0
        total_liabilities = row.get(f"{prefix}total_liabilities", 0) or 0
        current_liabilities = row.get(f"{prefix}current_liabilities", 0) or 0
        long_term_debt = row.get(f"{prefix}long_term_debt", 0) or 0
        
        # Calculate NCOA = AT - CA - IVAO
        # If IVAO is missing, set to zero (as per the table note)
        non_current_operating_assets = total_assets - current_assets - investment_advances
        
        # Calculate NCOL = LT - CL - DLTT
        # If DLTT is missing, set to zero (as per the table note)
        non_current_operating_liabilities = total_liabilities - current_liabilities - long_term_debt
        
        # Calculate Net NCOA = NCOA - NCOL
        net_non_current_operating_assets = non_current_operating_assets - non_current_operating_liabilities
        
        if total_assets is None or total_assets <= 0:
            return None
        return net_non_current_operating_assets / total_assets


class NetOperatingAssetsOverAssetsGrowthCalculator(GrowthCalculator):
    """Net Operating Assets over Total Assets growth with flexible lookback periods
    
    According to the Compustat definitions provided:
    - Operating Assets (OA) = COA + NCOA
    - Current Operating Assets (COA) = CA - CHE
    - Non-Current Operating Assets (NCOA) = AT - CA - IVAO
    - Operating Liabilities (OL) = COL + NCOL
    - Current Operating Liabilities (COL) = CL - DLC
    - Non-Current Operating Liabilities (NCOL) = LT - CL - DLTT
    - Net Operating Assets (NOA) = OA - OL
    """
    def __init__(self, lookback_quarters=4, use_pct_change=False):
        super().__init__(
            factor_name=f"noa_{lookback_quarters}q_growth",
            lookback_quarters=lookback_quarters,
            metric_name="NOA",
            use_pct_change=use_pct_change
        )

    def _get_financial_columns(self):
        return """
            cd.atq as current_assets,
            cd.actq as current_current_assets,
            cd.cheq as current_cash,
            cd.lctq as current_current_liabilities,
            cd.dlcq as current_short_term_debt,
            cd.ltq as current_total_liabilities,
            cd.dlttq as current_long_term_debt,
            cd.ivaoq as current_investment_advances_other,
            hd.atq as historical_assets,
            hd.actq as historical_current_assets,
            hd.cheq as historical_cash,
            hd.lctq as historical_current_liabilities,
            hd.dlcq as historical_short_term_debt,
            hd.ltq as historical_total_liabilities,
            hd.dlttq as historical_long_term_debt,
            hd.ivaoq as historical_investment_advances_other
        """

    def calculate_metric(self, row, period):
        prefix = f"{period}_"
        
        # Get all required variables
        total_assets = row.get(f"{prefix}assets")
        current_assets = row.get(f"{prefix}current_assets", 0) or 0
        cash = row.get(f"{prefix}cash", 0) or 0
        current_liabilities = row.get(f"{prefix}current_liabilities", 0) or 0
        short_term_debt = row.get(f"{prefix}short_term_debt", 0) or 0
        total_liabilities = row.get(f"{prefix}total_liabilities", 0) or 0
        long_term_debt = row.get(f"{prefix}long_term_debt", 0) or 0
        investment_advances = row.get(f"{prefix}investment_advances_other", 0) or 0
        
        # Calculate components according to definitions
        # Current Operating Assets (COA) = CA - CHE
        current_operating_assets = current_assets - cash
        
        # Non-Current Operating Assets (NCOA) = AT - CA - IVAO
        # If IVAO is missing, set to zero (per table note)
        noncurrent_operating_assets = total_assets - current_assets - investment_advances
        
        # Operating Assets (OA) = COA + NCOA
        operating_assets = current_operating_assets + noncurrent_operating_assets
        
        # Current Operating Liabilities (COL) = CL - DLC
        # If DLC is missing, set to zero (per table note)
        current_operating_liabilities = current_liabilities - short_term_debt
        
        # Non-Current Operating Liabilities (NCOL) = LT - CL - DLTT
        # If DLTT is missing, set to zero (per table note)
        noncurrent_operating_liabilities = total_liabilities - current_liabilities - long_term_debt
        
        # Operating Liabilities (OL) = COL + NCOL
        operating_liabilities = current_operating_liabilities + noncurrent_operating_liabilities
        
        # Net Operating Assets (NOA) = OA - OL
        net_operating_assets = operating_assets - operating_liabilities
        
        if total_assets is None or total_assets <= 0:
            return None
        return net_operating_assets / total_assets

class CashBasedOperatingProfitabilityCalculator(FactorCalculator):
    """
    Calculator for Cash-Based Operating Profitability (COP) scaled by assets.
    
    According to the definition:
    COP = (EBITDA* + XRD - OACC*) / Total Assets
    
    Where:
    - EBITDA* = NIQ + DPQ + XINTQ + TXTQ (Net Income + Depreciation + Interest + Taxes)
    - XRD = XRDQ (Research & Development expense, set to zero if unavailable)
    - OACC* = NIQ - OANCFQ (Net Income minus Operating Cash Flow)
    
    This measure is scaled by total assets to get the ratio.
    """
    def __init__(self):
        super().__init__("cash_based_op_profitability")
    
    def calculate(self, date_str, universe_df, duck_conn):
        import pandas as pd
        import numpy as np
        
        duck_conn.register('temp_universe', universe_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid']])
        
        query = f"""
        WITH current_year_data AS (
            SELECT 
                c.*,
                LAG(c.oancfy) OVER (PARTITION BY c.gvkey ORDER BY c.datadate) as prev_quarter_ytd_ocf,
                ROW_NUMBER() OVER (PARTITION BY c.gvkey ORDER BY c.datadate DESC) as qrank
            FROM wrds_csq_pit c
            INNER JOIN (
                SELECT DISTINCT gvkey FROM temp_universe
            ) u ON c.gvkey = u.gvkey
            WHERE 
                c.datadate <= DATE '{date_str}'
                AND c.indfmt = 'INDL'
                AND c.datafmt = 'STD'
                AND c.consol = 'C'
                AND c.popsrc = 'D'
                AND c.curcdq = 'USD'
                AND c.updq = 3
            QUALIFY qrank <= 5  -- Get current quarter + previous quarters for YTD calc
        ),
        current_quarter AS (
            SELECT *,
            -- Calculate quarterly OCF from YTD values
            CASE 
                WHEN fqtr = 1 THEN oancfy  -- Q1: Use YTD directly
                WHEN prev_quarter_ytd_ocf IS NULL THEN oancfy  -- No previous quarter data
                ELSE oancfy - prev_quarter_ytd_ocf  -- Q2-Q4: Current YTD minus previous quarter YTD
            END as quarterly_ocf
            FROM current_year_data
            WHERE qrank = 1  -- Only use the most recent quarter
        )
        SELECT 
            -- Select a single PERMNO record per GVKEY to avoid duplicates
            t.permno, t.lpermno, t.lpermco, t.gvkey, t.iid,
            cq.datadate,
            cq.atq as total_assets,
            
            -- Components needed for EBITDA* calculation
            cq.niq as net_income,
            cq.dpq as depreciation,
            cq.xintq as interest_expense,
            cq.txtq as income_taxes,
            
            -- Research & Development expense
            cq.xrdq as rd_expense,
            
            -- Operating cash flow (quarterly)
            cq.quarterly_ocf
            
        FROM (
            -- For each GVKEY, select only ONE corresponding PERMNO record
            SELECT DISTINCT ON (gvkey) permno, lpermno, lpermco, gvkey, iid
            FROM temp_universe
        ) t
        LEFT JOIN current_quarter cq ON t.gvkey = cq.gvkey
        """
        
        try:
            result_df = duck_conn.execute(query).fetchdf()
            
            # Make sure we don't have duplicates
            if len(result_df) > 0 and 'permno' in result_df.columns:
                result_df = result_df.drop_duplicates(subset=['permno'])
            
            # Calculate EBITDA* = NIQ + DPQ + XINTQ + TXTQ
            result_df['ebitda_star'] = result_df.apply(
                lambda x: x['net_income'] + 
                         (x['depreciation'] if pd.notnull(x['depreciation']) else 0) +
                         (x['interest_expense'] if pd.notnull(x['interest_expense']) else 0) +
                         (x['income_taxes'] if pd.notnull(x['income_taxes']) else 0)
                if pd.notnull(x['net_income']) else None,
                axis=1
            )
            
            # Handle R&D expense (set to 0 if unavailable)
            result_df['rd_expense_adj'] = result_df['rd_expense'].fillna(0)
            
            # Calculate Operating Accruals (OACC*) = NIQ - OANCFQ
            result_df['operating_accruals'] = result_df.apply(
                lambda x: x['net_income'] - x['quarterly_ocf']
                if (pd.notnull(x['net_income']) and pd.notnull(x['quarterly_ocf']))
                else None,
                axis=1
            )
            
            # Calculate Cash-Based Operating Profitability
            # COP = (EBITDA* + XRD - OACC*) / Total Assets
            result_df[self.factor_name] = result_df.apply(
                lambda x: (x['ebitda_star'] + x['rd_expense_adj'] - x['operating_accruals']) / x['total_assets']
                if (pd.notnull(x['ebitda_star']) and pd.notnull(x['operating_accruals']) and 
                    pd.notnull(x['total_assets']) and x['total_assets'] > 0)
                else None,
                axis=1
            )
            
            self._print_summary_statistics(result_df, date_str)
            
            return result_df[['permno', 'lpermno', 'lpermco', 'gvkey', 'iid', 
                             'datadate', self.factor_name]].rename(
                                 columns={'datadate': 'date'})
        finally:
            duck_conn.unregister('temp_universe')
    
    def _print_summary_statistics(self, df, date_str):
        if df.empty:
            print(f"\nNo valid Cash-Based Operating Profitability results for {date_str}")
            return
            
        total_stocks = len(df)
        valid_calcs = df[self.factor_name].notna().sum()
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Stocks with valid calculations: {valid_calcs} out of {total_stocks} ({valid_calcs/total_stocks*100:.1f}%)")
        
        print("\nCash-Based Operating Profitability Distribution:")
        print(df[self.factor_name].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
        
        # Print components distribution
        for col in ['ebitda_star', 'rd_expense_adj', 'operating_accruals']:
            if col in df.columns:
                print(f"\n{col} Distribution:")
                print(df[col].describe(percentiles=[.05, .25, .5, .75, .95]))
                
        # Check extreme values
        print("\nTop 5 extreme values:")
        print(df.nlargest(5, self.factor_name)[[self.factor_name, 'permno', 'gvkey']])
        print("\nBottom 5 extreme values:")
        print(df.nsmallest(5, self.factor_name)[[self.factor_name, 'permno', 'gvkey']])

class MonthlyResidualFF6MomentumCalculator(FactorCalculator):
    """
    Calculator for residual momentum based on Fama-French 6-factor model using monthly data.
    
    This implementation extends the FF3 residual momentum by including three additional factors:
    - CMA (Conservative Minus Aggressive)
    - RMW (Robust Minus Weak)
    - UMD (Momentum)
    
    The methodology follows the same approach:
    1. Factor loadings (betas) are estimated over a formation period using monthly returns
    2. These factor loadings are then used to calculate residual returns over a momentum period
    3. Residual momentum is calculated as the product of (1+residual_return)
    
    Parameters:
    -----------
    formation_months: Number of months to use for estimating factor loadings (e.g., 36 months)
    momentum_months: Number of months to use for calculating momentum (e.g., 11 months)
    skip_months: Number of months to skip between formation and momentum periods (default: 1)
    min_observations_formation: Minimum monthly observations required in formation period
    min_observations_momentum: Minimum monthly observations required in momentum period
    """
    def __init__(self, formation_months=36, momentum_months=11, skip_months=1, 
                 min_observations_formation=24, min_observations_momentum=8):
        factor_name = f"monthly_res_ff6_mom_f{formation_months}m_m{momentum_months}m"
        if skip_months > 0:
            factor_name += f"_s{skip_months}m"
        
        super().__init__(factor_name)
        self.formation_months = formation_months
        self.momentum_months = momentum_months
        self.skip_months = skip_months
        self.min_observations_formation = min_observations_formation
        self.min_observations_momentum = min_observations_momentum
        self.factor_name = factor_name
    
    def calculate(self, date_str, universe_df, duck_conn):
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm
        
        # Calculate date ranges
        end_date = pd.to_datetime(date_str)
        
        # Calculate the first day of current month
        current_month_start = pd.Timestamp(year=end_date.year, month=end_date.month, day=1)
        
        # Calculate the first day of the momentum period by subtracting momentum_months from current month
        momentum_start = current_month_start - pd.DateOffset(months=self.momentum_months)
        
        # Calculate the end of formation period (with skip if any)
        if self.skip_months > 0:
            formation_end = momentum_start - pd.DateOffset(months=self.skip_months)
        else:
            formation_end = momentum_start
        
        # Calculate the start of formation period
        formation_start = formation_end - pd.DateOffset(months=self.formation_months)
        
        # Convert dates to strings for SQL
        end_date_str = end_date.strftime('%Y-%m-%d')
        momentum_start_str = momentum_start.strftime('%Y-%m-%d')
        formation_end_str = formation_end.strftime('%Y-%m-%d')
        formation_start_str = formation_start.strftime('%Y-%m-%d')
        
        # Register universe DataFrame
        duck_conn.register('temp_universe', universe_df)
        
        # 1. Fetch daily data for formation period and convert to monthly returns
        # Include all 6 factors: mktrf, smb, hml, cma, rmw, umd
        formation_query = f"""
        WITH daily_returns AS (
            SELECT 
                sr.permno,
                sr.dlycaldt as date,
                sr.DlyRet,
                ff.mktrf,
                ff.smb,
                ff.hml,
                ff.cma,
                ff.rmw, 
                ff.umd,
                ff.rf,
                (sr.DlyRet - ff.rf) as excess_return,
                (1 + ff.rf) as rf_return,
                (1 + ff.mktrf) as mktrf_return,
                (1 + ff.smb) as smb_return,
                (1 + ff.hml) as hml_return,
                (1 + ff.cma) as cma_return,
                (1 + ff.rmw) as rmw_return,
                (1 + ff.umd) as umd_return,
                DATE_TRUNC('month', sr.dlycaldt) as month_start
            FROM stkdlysecuritydata sr
            INNER JOIN temp_universe u ON sr.permno = u.permno
            INNER JOIN fivefactors_daily ff ON sr.dlycaldt = ff.date
            WHERE sr.dlycaldt >= DATE '{formation_start_str}'
              AND sr.dlycaldt < DATE '{formation_end_str}'
              AND sr.DlyRet IS NOT NULL
        ),
        monthly_returns AS (
            SELECT 
                permno,
                month_start,
                (PRODUCT(1 + DlyRet) - 1) as monthly_ret,
                (PRODUCT(1 + excess_return) - 1) as monthly_excess_ret,
                (PRODUCT(mktrf_return) - 1) as monthly_mktrf,
                (PRODUCT(smb_return) - 1) as monthly_smb,
                (PRODUCT(hml_return) - 1) as monthly_hml,
                (PRODUCT(cma_return) - 1) as monthly_cma,
                (PRODUCT(rmw_return) - 1) as monthly_rmw,
                (PRODUCT(umd_return) - 1) as monthly_umd,
                (PRODUCT(rf_return) - 1) as monthly_rf,
                COUNT(*) as days_in_month
            FROM daily_returns
            GROUP BY permno, month_start
            HAVING COUNT(*) >= 15  -- Require at least 15 trading days in a month
        )
        SELECT * FROM monthly_returns
        ORDER BY permno, month_start
        """
        
        # 2. Fetch daily data for momentum period and convert to monthly returns
        momentum_query = f"""
        WITH daily_returns AS (
            SELECT 
                sr.permno,
                sr.dlycaldt as date,
                sr.DlyRet,
                ff.mktrf,
                ff.smb,
                ff.hml,
                ff.cma,
                ff.rmw,
                ff.umd,
                ff.rf,
                (sr.DlyRet - ff.rf) as excess_return,
                (1 + ff.rf) as rf_return,
                (1 + ff.mktrf) as mktrf_return,
                (1 + ff.smb) as smb_return,
                (1 + ff.hml) as hml_return,
                (1 + ff.cma) as cma_return,
                (1 + ff.rmw) as rmw_return,
                (1 + ff.umd) as umd_return,
                DATE_TRUNC('month', sr.dlycaldt) as month_start
            FROM stkdlysecuritydata sr
            INNER JOIN temp_universe u ON sr.permno = u.permno
            INNER JOIN fivefactors_daily ff ON sr.dlycaldt = ff.date
            WHERE sr.dlycaldt >= DATE '{momentum_start_str}'
              AND sr.dlycaldt <= DATE '{end_date_str}'
              AND sr.DlyRet IS NOT NULL
        ),
        monthly_returns AS (
            SELECT 
                permno,
                month_start,
                (PRODUCT(1 + DlyRet) - 1) as monthly_ret,
                (PRODUCT(1 + excess_return) - 1) as monthly_excess_ret,
                (PRODUCT(mktrf_return) - 1) as monthly_mktrf,
                (PRODUCT(smb_return) - 1) as monthly_smb,
                (PRODUCT(hml_return) - 1) as monthly_hml,
                (PRODUCT(cma_return) - 1) as monthly_cma,
                (PRODUCT(rmw_return) - 1) as monthly_rmw,
                (PRODUCT(umd_return) - 1) as monthly_umd,
                (PRODUCT(rf_return) - 1) as monthly_rf,
                COUNT(*) as days_in_month
            FROM daily_returns
            GROUP BY permno, month_start
            HAVING COUNT(*) >= 15  -- Require at least 15 trading days in a month
        )
        SELECT * FROM monthly_returns
        ORDER BY permno, month_start
        """
        
        # Execute queries
        formation_df = duck_conn.execute(formation_query).fetchdf()
        momentum_df = duck_conn.execute(momentum_query).fetchdf()
        
        # Estimate factor loadings and calculate residual momentum
        results = []
        
        # Get unique permnos
        permnos = universe_df['permno'].unique()
        
        for permno in permnos:
            try:
                # Get formation data for this stock
                formation_stock = formation_df[formation_df['permno'] == permno]
                
                # Only proceed if we have enough monthly observations
                if len(formation_stock) < self.min_observations_formation:
                    continue
                
                # Estimate factor loadings using formation period
                # Use all 6 factors in the regression
                y = formation_stock['monthly_excess_ret']
                X = formation_stock[['monthly_mktrf', 'monthly_smb', 'monthly_hml', 
                                    'monthly_cma', 'monthly_rmw', 'monthly_umd']]
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                
                # Get momentum period data
                momentum_stock = momentum_df[momentum_df['permno'] == permno]
                
                # Only proceed if we have enough monthly observations
                if len(momentum_stock) < self.min_observations_momentum:
                    continue
                
                # Calculate residual returns for momentum period
                X_momentum = momentum_stock[['monthly_mktrf', 'monthly_smb', 'monthly_hml',
                                           'monthly_cma', 'monthly_rmw', 'monthly_umd']]
                X_momentum = sm.add_constant(X_momentum)
                
                # Expected returns based on factor loadings
                expected_returns = X_momentum.dot(model.params)
                
                # Residual returns
                residual_returns = momentum_stock['monthly_excess_ret'] - expected_returns
                
                # Calculate residual momentum as product of (1+residual_return)
                res_momentum = np.prod(1 + residual_returns) - 1
                
                # Store results
                results.append({
                    'permno': permno,
                    'formation_obs': len(formation_stock),
                    'momentum_obs': len(momentum_stock),
                    'alpha': model.params['const'],
                    'beta_mkt': model.params['monthly_mktrf'],
                    'beta_smb': model.params['monthly_smb'],
                    'beta_hml': model.params['monthly_hml'],
                    'beta_cma': model.params['monthly_cma'],
                    'beta_rmw': model.params['monthly_rmw'],
                    'beta_umd': model.params['monthly_umd'],
                    self.factor_name: res_momentum
                })
                
            except Exception as e:
                print(f"Error processing permno {permno}: {str(e)}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Return results with all stocks in universe
        if len(results_df) > 0:
            duck_conn.register('res_momentum_results', results_df)
            
            final_query = f"""
            SELECT 
                u.permno,
                u.lpermno,
                u.lpermco,
                u.gvkey,
                u.iid,
                DATE '{date_str}' as date,
                r.{self.factor_name}
            FROM temp_universe u
            LEFT JOIN res_momentum_results r ON u.permno = r.permno
            """
            
            final_result = duck_conn.execute(final_query).fetchdf()
        else:
            # If no results, return just universe with missing factor values
            final_result = universe_df.copy()
            final_result['date'] = pd.to_datetime(date_str)
            final_result[self.factor_name] = np.nan
        
        # Clean up
        duck_conn.unregister('temp_universe')
        if len(results_df) > 0:
            duck_conn.unregister('res_momentum_results')
        
        self._print_summary_statistics(results_df, date_str)
        
        return final_result
    
    def _print_summary_statistics(self, df, date_str):
        if df.empty:
            print(f"\nNo valid monthly residual FF6 momentum results for {date_str}")
            return
            
        total_stocks = len(df)
        
        print(f"\n{self.factor_name} Summary for {date_str}:")
        print(f"Stocks with valid monthly residual FF6 momentum: {total_stocks}")
        
        print("\nMonthly Residual Momentum Distribution:")
        print(df[self.factor_name].describe(percentiles=[.05, .25, .5, .75, .95]))
        
        print("\nFactor Loadings from Formation Period:")
        for col in ['alpha', 'beta_mkt', 'beta_smb', 'beta_hml', 'beta_cma', 'beta_rmw', 'beta_umd']:
            print(f"\n{col} statistics:")
            print(df[col].describe(percentiles=[.05, .25, .5, .75, .95]))
                
        print("\nObservation Counts:")
        print("Formation Period (months):")
        print(df['formation_obs'].describe(percentiles=[.05, .25, .5, .75, .95]))
        print("\nMomentum Period (months):")
        print(df['momentum_obs'].describe(percentiles=[.05, .25, .5, .75, .95]))