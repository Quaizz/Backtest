
# FactorCalculator.py

import pandas as pd
import os
from pathlib import Path

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