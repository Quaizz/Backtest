
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
    Calculator for Book-to-Market ratio (BP).
    BP = 1 / PB Ratio = Book Value / Market Cap
    
    This is the inverse of Price-to-Book ratio and is often used as a value metric.
    
    Compustat fields:
    - Book Value: ceqq (common equity)
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
            q.book_value,
            s.dlycap as market_cap,
            -- Calculate Price-to-Book ratio
            CASE 
                WHEN q.book_value <= 0 THEN NULL
                ELSE s.dlycap / q.book_value
            END as price_to_book,
            -- Calculate Book-to-Price ratio (inverse of P/B)
            CASE 
                WHEN s.dlycap <= 0 THEN NULL
                ELSE q.book_value / s.dlycap
            END as book_to_price
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