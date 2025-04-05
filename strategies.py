# strategies.py
import pandas as pd
import numpy as np
from pathlib import Path
import os

class PortfolioStrategy:
    """Base class for portfolio generation strategies"""
    
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
    
    def generate_portfolio_targets(self, date, data_df, min_price=2.0, market_cap_percentile=0.5, stock_num=10, logger=print):
        """
        Generate target portfolio based on strategy logic
        
        Parameters:
        -----------
        date : datetime
            Current rebalancing date
        data_df : DataFrame
            Daily stock data including factors
        min_price : float
            Minimum price threshold for stocks
        market_cap_percentile : float
            Market cap percentile threshold (0-1)
        stock_num : int
            Number of stocks to select
        logger : function
            Logging function
            
        Returns:
        --------
        dict : {symbol: target_weight}
        """
        raise NotImplementedError("Subclasses must implement generate_portfolio_targets")


class MultiFactorStrategy1(PortfolioStrategy):
    """Strategy with simplified ranking logic using direct rank columns"""
    
    def __init__(self, industry_limit=2, store_intermediate=True):
        """
        Initialize strategy
        
        Parameters:
        -----------
        industry_limit : int
            Maximum number of stocks per industry
        store_intermediate : bool
            Whether to store intermediate DataFrames
        """
        # Define the factors required by this strategy
        factor_list = [
            # Quality factors
            'cash_flow_to_asset',
            'cash_flow_to_liability',
            'total_asset_turnover_rate',
            
            # Growth factors
            'gpoa_4q_growth',
            'roe',
            'roic',
            
            # Value and momentum factors
            'earnings_to_price',
            'cum_return_252d_offset_21d',
            'cum_return_126d_offset_21d',
            'cum_return_21d'
        ]
        
        # Call parent constructor with factor list
        super().__init__(factor_list)
        
        # Store other parameters
        self.industry_limit = industry_limit
        self.store_intermediate = store_intermediate
        self.intermediate_dfs = {}  # Dictionary to store DataFrames at different stages

    def generate_portfolio_targets(self, date, data_df, min_price=2.0, market_cap_percentile=0.5, stock_num=50, logger=print):
        """Generate portfolio targets using simplified approach"""
    
        # Reset intermediate storage for this run
        if self.store_intermediate:
            self.intermediate_dfs = {}

        # Validate data
        if data_df is None or len(data_df) == 0:
            logger("Error: Empty dataframe passed to generate_portfolio_targets")
            return {}

        # Filter universe first
        universe_df = data_df[
            (data_df['dlyprc'] > min_price) &
            (data_df['dlycap'] > data_df['dlycap'].quantile(market_cap_percentile))
        ].copy()
        
        logger(f"Starting universe size: {len(universe_df)} stocks")
        
        # Store initial filtered universe
        if self.store_intermediate:
            self.intermediate_dfs['initial_universe'] = universe_df.copy()

        # STAGE 1: Quality Factors Ranking with direct manual ranking
        logger("Stage 1: Ranking on quality factors")
        
        # Direct manual ranking for each quality factor
        universe_df['cash_flow_to_asset_rank'] = universe_df['cash_flow_to_asset'].rank(ascending=True)
        universe_df['cash_flow_to_liability_rank'] = universe_df['cash_flow_to_liability'].rank(ascending=True)
        universe_df['total_asset_turnover_rate_rank'] = universe_df['total_asset_turnover_rate'].rank(ascending=True)
        
        # Calculate point1 as sum of all quality factor ranks
        universe_df['point1'] = universe_df['cash_flow_to_asset_rank'] + universe_df['cash_flow_to_liability_rank'] + universe_df['total_asset_turnover_rate_rank']
        
        # Sort and keep top 2/3
        universe_df.sort_values(by='point1', ascending=False, inplace=True)
        universe_df = universe_df.iloc[:len(universe_df)//3*2]
        
        logger(f"After quality filter: {len(universe_df)} stocks")

        # Store post-quality filter universe
        if self.store_intermediate:
            self.intermediate_dfs['post_quality_filter'] = universe_df.copy()
        
        # Early exit if no stocks remain
        if len(universe_df) == 0:
            logger("No stocks remain after Stage 1 filtering")
            return {}
        
        # STAGE 2: Growth Factors Ranking with direct manual ranking
        logger("Stage 2: Ranking on growth factors")
        
        # Direct manual ranking for each growth factor
        universe_df['gpoa_4q_growth_rank'] = universe_df['gpoa_4q_growth'].rank(ascending=True)
        universe_df['roe_rank'] = universe_df['roe'].rank(ascending=True)
        universe_df['roic_rank'] = universe_df['roic'].rank(ascending=True)
        
        # Calculate point3 as sum of all growth factor ranks
        universe_df['point3'] = universe_df['gpoa_4q_growth_rank'] + universe_df['roe_rank'] + universe_df['roic_rank']
        
        # Sort and keep top 1/4
        universe_df.sort_values(by='point3', ascending=False, inplace=True)
        universe_df = universe_df.iloc[:len(universe_df)//4]
        
        logger(f"After growth filter: {len(universe_df)} stocks")
        
        # Store post-growth filter universe
        if self.store_intermediate:
            self.intermediate_dfs['post_growth_filter'] = universe_df.copy()
            
        # Early exit if no stocks remain
        if len(universe_df) == 0:
            logger("No stocks remain after Stage 2 filtering")
            return {}
        
        # STAGE 3: Value and Momentum Factors - Direct rank definitions
        logger("Stage 3: Using value and momentum factors")
        
        # Add rank columns with direct definitions (no loops)
        universe_df['earnings_to_price_rank'] = universe_df['earnings_to_price'].rank(ascending=True)
        universe_df['cum_return_252d_offset_21d_rank'] = universe_df['cum_return_252d_offset_21d'].rank(ascending=True)
        universe_df['cum_return_126d_offset_21d_rank'] = universe_df['cum_return_126d_offset_21d'].rank(ascending=True)
        
        # Sum up all rank columns to get point8
        universe_df['point8'] = universe_df['earnings_to_price_rank'] + universe_df['cum_return_252d_offset_21d_rank'] + universe_df['cum_return_126d_offset_21d_rank']
        
        # Sort by point8 (higher is better)
        universe_df.sort_values(by='point3', ascending=False, inplace=True)
        
        logger(f"After stage 3 filtering: ready for final selection")
        
        # Store final ranked universe
        if self.store_intermediate:
            self.intermediate_dfs['final_ranked'] = universe_df.copy()
        
        # Select final stocks (top N stocks)
        selected_stocks = universe_df.head(stock_num).index
        
        # Store final selected universe
        if self.store_intermediate:
            self.intermediate_dfs['final_selection'] = universe_df.head(stock_num).copy()
        
        # Equal weight portfolio
        weight = 1.0 / len(selected_stocks) if len(selected_stocks) > 0 else 0
        target_portfolio = {int(stock): weight for stock in selected_stocks}
        
        self.save_intermediate_dataframes(date)
        # Log portfolio construction details
        logger(f"Final portfolio: {len(target_portfolio)} stocks selected")
        return target_portfolio
    
    def get_intermediate_dataframes(self):
        """
        Return the stored intermediate DataFrames
        
        Returns:
        --------
        dict : {stage_name: dataframe}
        """
        if not self.store_intermediate:
            return {"error": "Intermediate DataFrame storage is not enabled. Initialize with store_intermediate=True"}
        return self.intermediate_dfs
        
    def save_intermediate_dataframes(self, date, directory="intermediate_dfs"):
        """
        Save all intermediate DataFrames to CSV files
        
        Parameters:
        -----------
        date : str
            Date string for file naming
        directory : str
            Directory to save files in
        """
        import os
        if not self.store_intermediate:
            print("Intermediate DataFrame storage is not enabled. Initialize with store_intermediate=True")
            return
            
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        date_str = date.strftime('%Y%m%d')
        # Save each DataFrame
        for stage, df in self.intermediate_dfs.items():
            filename = f"{directory}/{date_str}_{stage}.csv"
            df.to_csv(filename)




class MultiFactorStrategy2(PortfolioStrategy):
    """Strategy with simplified ranking logic using direct rank columns"""
    
    def __init__(self, industry_limit=2, store_intermediate=True):
        """
        Initialize strategy
        
        Parameters:
        -----------
        industry_limit : int
            Maximum number of stocks per industry
        store_intermediate : bool
            Whether to store intermediate DataFrames
        """
        # Define the factors required by this strategy
        factor_list = [
            # Quality factors
            'cash_flow_to_asset',
            'cash_flow_to_liability',
            'total_asset_turnover_rate',
            
            # Growth factors
            'gpoa_4q_growth',
            'roe',
            'roic',
            
            # Value and momentum factors
            'earnings_to_price',
            'cum_return_252d_offset_21d',
            'cum_return_126d_offset_21d',
            'cum_return_21d',
            
            'MAX5_21d',
            'inv_4q_growth',
            'cash_based_op_profitability',

        ]
        
        # Call parent constructor with factor list
        super().__init__(factor_list)
        
        # Store other parameters
        self.industry_limit = industry_limit
        self.store_intermediate = store_intermediate
        self.intermediate_dfs = {}  # Dictionary to store DataFrames at different stages

    def generate_portfolio_targets(self, date, data_df, min_price=2.0, market_cap_percentile=0.5, stock_num=50, logger=print):
        """Generate portfolio targets using simplified approach"""
    
        # Reset intermediate storage for this run
        if self.store_intermediate:
            self.intermediate_dfs = {}

        # Validate data
        if data_df is None or len(data_df) == 0:
            logger("Error: Empty dataframe passed to generate_portfolio_targets")
            return {}

        # Filter universe first
        universe_df = data_df[
            (data_df['dlyprc'] > min_price) &
            (data_df['dlycap'] > data_df['dlycap'].quantile(market_cap_percentile))
        ].copy()
        
        logger(f"Starting universe size: {len(universe_df)} stocks")
        
        # Store initial filtered universe
        if self.store_intermediate:
            self.intermediate_dfs['initial_universe'] = universe_df.copy()

        # STAGE 1: Quality Factors Ranking with direct manual ranking
        logger("Stage 1: Ranking on quality factors")
        
        # Direct manual ranking for each quality factor
        universe_df['cash_flow_to_asset_rank'] = universe_df['cash_flow_to_asset'].rank(ascending=True)
        universe_df['cash_flow_to_liability_rank'] = universe_df['cash_flow_to_liability'].rank(ascending=True)
        universe_df['total_asset_turnover_rate_rank'] = universe_df['total_asset_turnover_rate'].rank(ascending=True)
        universe_df['earnings_to_price_rank'] = universe_df['earnings_to_price'].rank(ascending=True)
        universe_df['cum_return_252d_offset_21d_rank'] = universe_df['cum_return_252d_offset_21d'].rank(ascending=True)
        universe_df['cum_return_126d_offset_21d_rank'] = universe_df['cum_return_126d_offset_21d'].rank(ascending=True)


        # Calculate point1 as sum of all quality factor ranks
        universe_df['point1'] = universe_df['cash_flow_to_asset_rank'] + universe_df['cash_flow_to_liability_rank'] + universe_df['total_asset_turnover_rate_rank']+universe_df['earnings_to_price_rank']+universe_df['cum_return_252d_offset_21d_rank']+universe_df['cum_return_126d_offset_21d_rank']
        
        # Sort and keep top 2/3
        universe_df.sort_values(by='point1', ascending=False, inplace=True)
        universe_df = universe_df.iloc[:len(universe_df)//3*2]
        
        logger(f"After quality filter: {len(universe_df)} stocks")

        # Store post-quality filter universe
        if self.store_intermediate:
            self.intermediate_dfs['post_quality_filter'] = universe_df.copy()
        
        # Early exit if no stocks remain
        if len(universe_df) == 0:
            logger("No stocks remain after Stage 1 filtering")
            return {}
        
        # STAGE 2: Growth Factors Ranking with direct manual ranking
        logger("Stage 2: Ranking on growth factors")
        
        # Direct manual ranking for each growth factor
        universe_df['gpoa_4q_growth_rank'] = universe_df['gpoa_4q_growth'].rank(ascending=True)
        universe_df['roe_rank'] = universe_df['roe'].rank(ascending=True)
        universe_df['roic_rank'] = universe_df['roic'].rank(ascending=True)

        universe_df['cash_based_op_profitability_rank'] = universe_df['cash_based_op_profitability'].rank(ascending=True)
        universe_df['MAX5_21d_rank'] = universe_df['MAX5_21d'].rank(ascending=False)
        universe_df['inv_4q_growth_rank'] = universe_df['inv_4q_growth'].rank(ascending=False)
        universe_df['cum_return_252d_offset_21d_rank'] = universe_df['cum_return_252d_offset_21d'].rank(ascending=True)


        # Calculate point3 as sum of all growth factor ranks
        universe_df['point3'] = universe_df['gpoa_4q_growth_rank'] + universe_df['roe_rank'] + universe_df['roic_rank']+ universe_df['cash_based_op_profitability_rank'] +universe_df['MAX5_21d_rank'] +universe_df['inv_4q_growth_rank']#+universe_df['cum_return_252d_offset_21d_rank']
        # Sort and keep top 1/4
        '''
        universe_df.sort_values(by='point3', ascending=False, inplace=True)
        universe_df = universe_df.iloc[:len(universe_df)//4]
        
        logger(f"After growth filter: {len(universe_df)} stocks")
        
        # Store post-growth filter universe
        if self.store_intermediate:
            self.intermediate_dfs['post_growth_filter'] = universe_df.copy()
            
        # Early exit if no stocks remain
        if len(universe_df) == 0:
            logger("No stocks remain after Stage 2 filtering")
            return {}
        
        # STAGE 3: Value and Momentum Factors - Direct rank definitions
        logger("Stage 3: Using value and momentum factors")
        
        # Add rank columns with direct definitions (no loops)
        universe_df['earnings_to_price_rank'] = universe_df['earnings_to_price'].rank(ascending=True)
        universe_df['cum_return_252d_offset_21d_rank'] = universe_df['cum_return_252d_offset_21d'].rank(ascending=True)
        universe_df['cum_return_126d_offset_21d_rank'] = universe_df['cum_return_126d_offset_21d'].rank(ascending=True)

        universe_df['cash_based_op_profitability_rank'] = universe_df['cash_based_op_profitability'].rank(ascending=True)
        universe_df['MAX5_21d_rank'] = universe_df['MAX5_21d'].rank(ascending=False)
        universe_df['inv_4q_growth_rank'] = universe_df['inv_4q_growth'].rank(ascending=False)
        
        # Sum up all rank columns to get point8
        universe_df['point8'] = universe_df['earnings_to_price_rank'] + universe_df['cum_return_252d_offset_21d_rank'] + universe_df['cum_return_126d_offset_21d_rank']
        '''
        # Sort by point8 (higher is better)
        universe_df.sort_values(by='point3', ascending=False, inplace=True)
        
        logger(f"After stage 3 filtering: ready for final selection")
        
        # Store final ranked universe
        if self.store_intermediate:
            self.intermediate_dfs['final_ranked'] = universe_df.copy()
        
        # Select final stocks (top N stocks)
        selected_stocks = universe_df.head(stock_num).index
        
        # Store final selected universe
        if self.store_intermediate:
            self.intermediate_dfs['final_selection'] = universe_df.head(stock_num).copy()
        
        # Equal weight portfolio
        weight = 1.0 / len(selected_stocks) if len(selected_stocks) > 0 else 0
        target_portfolio = {int(stock): weight for stock in selected_stocks}
        
        self.save_intermediate_dataframes(date)
        # Log portfolio construction details
        logger(f"Final portfolio: {len(target_portfolio)} stocks selected")
        return target_portfolio
    
    def get_intermediate_dataframes(self):
        """
        Return the stored intermediate DataFrames
        
        Returns:
        --------
        dict : {stage_name: dataframe}
        """
        if not self.store_intermediate:
            return {"error": "Intermediate DataFrame storage is not enabled. Initialize with store_intermediate=True"}
        return self.intermediate_dfs
        
    def save_intermediate_dataframes(self, date, directory="intermediate_dfs"):
        """
        Save all intermediate DataFrames to CSV files
        
        Parameters:
        -----------
        date : str
            Date string for file naming
        directory : str
            Directory to save files in
        """
        import os
        if not self.store_intermediate:
            print("Intermediate DataFrame storage is not enabled. Initialize with store_intermediate=True")
            return
            
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        date_str = date.strftime('%Y%m%d')
        # Save each DataFrame
        for stage, df in self.intermediate_dfs.items():
            filename = f"{directory}/{date_str}_{stage}.csv"
            df.to_csv(filename)