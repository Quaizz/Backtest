import os
import numpy as np
import scipy.stats 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import warnings

class USQuantileFactorAnalysis:
    def __init__(self, data_df_dic,  factor_list, universe, 
                 group_num, trading_date, rebalance_freq, balance_day, ic_periods):
        """
        Initialize factor analysis using daily return data directly.
        
        Parameters:
        -----------
        data_df_dic : dict
            Dictionary of daily stock data including returns
        price : str
            Name of the price field (e.g., 'dlyprc')
        factor_list : list
            List of factors to analyze
        universe : str
            Universe identifier
        group_num : int
            Number of quantile groups
        trading_date : list
            List of trading dates
        rebalance_freq : str
            Rebalancing frequency ('1d', '1w', '1m')
        ic_periods : int
            Number of periods for IC calculation
        """
        self.data_df_dic = data_df_dic
        self.factor_list = factor_list
        self.universe = universe
        self.group_num = group_num
        self.trading_date = trading_date
        self.rebalance_freq = rebalance_freq
        self.ic_periods = ic_periods
        self.balance_day = balance_day
        # Calculate rebalancing dates first
        self.balance_date = self.check_trade_day(trading_date, rebalance_freq, balance_day)
        

        # Initialize storage for analysis results
        self.group_position_list = [{} for _ in range(group_num)]
        self.group_daily_return_list = [[] for _ in range(group_num)]
        
        # Performance metrics storage
        self.group_return_std_list = [[] for _ in range(group_num)]
        self.cum_list = []
        self.daily_turnover_list = [[] for _ in range(group_num)]
        
        # Results storage
        self.annual_returns = None
        self.sharpe_ratios = None
        self.max_drawdowns = None
        self.size_dic = {}

    def check_trade_day(self, trade_date_list, rebalance_freq, balance_day):
        """
        Determine rebalancing dates based on frequency and preferred day.
        
        Parameters:
        -----------
        trade_date_list : list
            List of all trading dates
        rebalance_freq : str
            Rebalancing frequency ('1d', '1w', '1m')
        balance_day : int
            Target day for rebalancing (0-6 for weekly, 1-31 for monthly)
            
        Returns:
        --------
        list
            List of dates when rebalancing should occur
        """
        # Always include the first trading date
        first_date = trade_date_list[1]
        
        # For daily rebalancing, use all trading dates
        if rebalance_freq == '1d':
            return trade_date_list
            
        # Convert dates to pandas datetime for easier manipulation
        dates_df = pd.DataFrame({
            'date_str': trade_date_list,
            'date': pd.to_datetime(trade_date_list)
        })
        
        if rebalance_freq == '1w':
            # Adjust balance_day to match desired weekday
            # (balance_day should be 0-6, where 0 is Monday)
            target_weekday = balance_day % 7
            
            # Find dates that match the target weekday
            rebalance_mask = dates_df['date'].dt.weekday == target_weekday
            rebalance_dates = dates_df[rebalance_mask]['date_str'].tolist()
            
        elif rebalance_freq == '1m':
            # Group dates by year-month
            dates_df['ym'] = dates_df['date'].dt.strftime('%Y%m')
            monthly_groups = dates_df.groupby('ym')
            
            rebalance_dates = []
            for _, group in monthly_groups:
                group = group.sort_values('date')
                
                # Get the last possible trading day for this target day
                target_day = min(balance_day, group['date'].iloc[-1].day)
                valid_dates = group[group['date'].dt.day <= target_day]
                
                if not valid_dates.empty:
                    # Get the last valid trading day before or on target day
                    rebalance_dates.append(valid_dates.iloc[-1]['date_str'])
                else:
                    # If no valid dates before target day, use month's first trading day
                    rebalance_dates.append(group.iloc[0]['date_str'])
                    
        else:
            raise ValueError(f"Invalid rebalance frequency: {rebalance_freq}")
            
        
        # Ensure first trading date is included and dates are properly ordered
        if first_date not in rebalance_dates:
            rebalance_dates = [first_date] + sorted(rebalance_dates)
        else:
            rebalance_dates = sorted(rebalance_dates)
            
        return rebalance_dates
        
    def initialize_portfolio(self, start_date):
        """
        Initialize the portfolio using factor data from the previous trading date.
        
        Parameters:
        -----------
        start_date : str
            The first trading date after the start date
        """
        # Get the previous trading date
        prev_date_idx = self.trading_date.index(start_date) - 1
        if prev_date_idx < 0:
            raise ValueError("No previous trading date available for initialization")
            
        prev_date = self.trading_date[prev_date_idx]
        prev_data_df = self.data_df_dic[prev_date]
        
        # Apply filters for investable universe
        factor_df = prev_data_df[
            (prev_data_df['dlycap'] > 500) &  # Market cap > $500M
            (prev_data_df['dlyprc'] > 5)      # Price > $5
        ]
        
        factor_df.dropna(subset=self.factor_list)
        
        # Calculate initial factor values
        self.cal_factor_func(factor_df)
        factor_df.sort_values(by='point1', ascending=False, inplace=True)
        
        # Initialize group positions
        total_stocks = len(factor_df)
        for group_i in range(self.group_num):
            start_idx = int(total_stocks * group_i / self.group_num)
            end_idx = int(total_stocks * (group_i + 1) / self.group_num)
            self.group_position_list[group_i] = factor_df.index[start_idx:end_idx].tolist()
            
        # Initialize other class members
        for group_i in range(self.group_num):
            self.group_daily_return_list[group_i] = []
            self.daily_turnover_list[group_i] = []    

    def process_daily_returns(self, returns_df):
        """
        Process daily returns according to CRSP return flags.
        """
        processed_df = returns_df.copy()
        
        # Drop stocks with critical flags
        drop_flags = ['NT', 'MP', 'RA']
        stocks_to_drop = processed_df[
            processed_df['dlyretmissflg'].isin(drop_flags)
        ].index.unique()
        
        if len(stocks_to_drop) > 0:
            processed_df = processed_df[~processed_df.index.isin(stocks_to_drop)]
        
        # Handle remaining missing returns
        missing_returns = processed_df['dlyret'].isna()
        if missing_returns.any():
            for flag_type in processed_df.loc[missing_returns, 'dlyretmissflg'].unique():
                flag_mask = (missing_returns) & (processed_df['dlyretmissflg'] == flag_type)
                
                if flag_type in ['DM', 'DG', 'DP']:
                    processed_df.loc[flag_mask, 'dlyret'] = -1.0
                    
                elif flag_type == 'GP':
                    processed_df.loc[flag_mask, 'dlyret'] = 0.0
                    
                elif flag_type == 'NS':
                    ns_mask = flag_mask & processed_df['dlyclose'].notna() & processed_df['dlyopen'].notna()
                    
                    if ns_mask.any():
                        processed_df.loc[ns_mask, 'dlyret'] = (
                            processed_df.loc[ns_mask, 'dlyclose'] / 
                            processed_df.loc[ns_mask, 'dlyopen'] - 1
                        )
                        
                        remaining_ns = flag_mask & ~ns_mask
                        if remaining_ns.any():
                            processed_df.loc[remaining_ns, 'dlyret'] = 0.0
        
        return processed_df

    def cal_factor_func(self, factor_df):
        """Calculate factor values - override for specific factors."""
        factor_df['point1'] = factor_df[self.factor_list[0]]


    def calculate_daily_returns(self, date, current_data_df):
        """Calculate daily returns for existing positions."""
        if len(self.group_position_list[0]) > 0:
            # Filter for valid stocks
            valid_stocks_mask = (
                (current_data_df['dlycap'] > 500) &
                (current_data_df['dlyprc'] > 5)
            )
            
            day_returns = current_data_df[valid_stocks_mask][
                ['dlyret', 'dlyretmissflg', 'dlyclose', 'dlyopen']
            ]
            
            processed_returns = self.process_daily_returns(day_returns)
            
            # Calculate returns for each group
            for group_i in range(self.group_num):
                group_stocks = self.group_position_list[group_i]
                if len(group_stocks) > 0:
                    valid_stocks = processed_returns.index.intersection(group_stocks)
                    if len(valid_stocks) > 0:
                        stock_values = processed_returns.loc[valid_stocks, 'dlyret'] + 1
                        mean_final_value = stock_values.mean()
                        group_return = mean_final_value - 1
                        self.group_daily_return_list[group_i].append(
                            float(group_return) if not np.isnan(group_return) else 0
                        )
                    else:
                        self.group_daily_return_list[group_i].append(0)
                else:
                    self.group_daily_return_list[group_i].append(0)
        else:
            for group_i in range(self.group_num):
                self.group_daily_return_list[group_i].append(0)
                
    def rebalance_portfolio(self, date):
        """Rebalance portfolio using previous day's data."""
        prev_date_idx = self.trading_date.index(date) - 1
        prev_date = self.trading_date[prev_date_idx]
        prev_data_df = self.data_df_dic[prev_date]
        
        # Apply filters for investable universe
        factor_df = prev_data_df[
            (prev_data_df['dlycap'] > 500) &  # Market cap > $500M
            (prev_data_df['dlyprc'] > 5)      # Price > $5
        ]
        
        factor_df.dropna(subset=self.factor_list)
        
        # Calculate factors and sort
        self.cal_factor_func(factor_df)
        factor_df.sort_values(by='point1', ascending=False, inplace=True)
        
        # Calculate new group assignments
        total_stocks = len(factor_df)
        for group_i in range(self.group_num):
            start_idx = int(total_stocks * group_i / self.group_num)
            end_idx = int(total_stocks * (group_i + 1) / self.group_num)
            new_group_stocks = factor_df.index[start_idx:end_idx].tolist()
            
            # Calculate turnover
            old_position = set(self.group_position_list[group_i])
            new_position = set(new_group_stocks)
            turnover = 1 - len(old_position & new_position) / len(old_position) if old_position else 0
            self.daily_turnover_list[group_i].append(turnover)
            
            # Update positions
            self.group_position_list[group_i] = new_group_stocks


    def _calculate_performance_metrics(self):
        """
        Calculate key performance metrics including:
        - Cumulative returns
        - Annual returns
        - Sharpe ratios
        - Maximum drawdowns
        
        This method updates the class attributes with calculated metrics.
        """
        # Calculate cumulative returns using the dates where we have returns
        self.cum_list = []
        for group_returns in self.group_daily_return_list:
            cum_returns = np.cumprod(1 + np.array(group_returns))
            self.cum_list.append(cum_returns.tolist())
        
        # Use 252 trading days per year for annualization
        trading_days_per_year = 252
        
        # Calculate annual returns
        self.annual_returns = []
        for group_returns in self.group_daily_return_list:
            total_return = (1 + np.array(group_returns)).prod() - 1  # Total return over period
            years = len(group_returns) / trading_days_per_year  # Number of years
            ann_return = ((1 + total_return) ** (1/years) - 1) * 100  # Annualized return
            self.annual_returns.append(ann_return)
        
        # Calculate Sharpe ratios
        self.sharpe_ratios = []
        for returns in self.group_daily_return_list:
            returns_array = np.array(returns)
            # Annualize mean and standard deviation
            mean_return = np.mean(returns_array) * trading_days_per_year
            std_return = np.std(returns_array) * np.sqrt(trading_days_per_year)
            sharpe = mean_return / std_return if std_return != 0 else 0
            self.sharpe_ratios.append(sharpe)
        
        # Calculate maximum drawdowns
        self.max_drawdowns = []
        for cum_rets in self.cum_list:
            cum_returns = np.array(cum_rets)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (running_max - cum_returns) / running_max
            max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
            self.max_drawdowns.append(max_drawdown)

    '''
    def balance_func(self, date, current_data_df):
        """
        Perform portfolio rebalancing and daily return calculations.
        Use previous day's data for rebalancing decisions to avoid look-ahead bias.
        """
        # Calculate daily returns for current groups regardless of rebalance
        if len(self.group_position_list[0]) > 0:  # If we have existing positions
            day_returns = current_data_df[
                ['dlyret', 'dlyretmissflg', 'dlyclose', 'dlyopen']
            ]
            
            processed_returns = self.process_daily_returns(day_returns)
            
            # Calculate returns for each group
            for group_i in range(self.group_num):
                group_stocks = self.group_position_list[group_i]
                if len(group_stocks) > 0:
                    valid_stocks = processed_returns.index.intersection(group_stocks)
                    if len(valid_stocks) > 0:
                        stock_values = processed_returns.loc[valid_stocks, 'dlyret'] + 1
                        mean_final_value = stock_values.mean()
                        group_return = mean_final_value - 1
                        self.group_daily_return_list[group_i].append(
                            float(group_return) if not np.isnan(group_return) else 0
                        )
                    else:
                        self.group_daily_return_list[group_i].append(0)
                else:
                    self.group_daily_return_list[group_i].append(0)

        # Rebalance portfolios only on balance dates using previous day's data
        if date in self.balance_date:
            prev_date_idx = self.trading_date.index(date) - 1
            if prev_date_idx >= 0:  # Make sure it's not the first date
                prev_date = self.trading_date[prev_date_idx]
                prev_data_df = self.data_df_dic[prev_date]  # Get previous day's data
                
                # First apply all data quality filters
                factor_df = prev_data_df.dropna(subset= self.factor_list)
                
                
                # Then apply the condition filters
                #factor_df = factor_df[
                #    (factor_df['tradingstatusflg'] == 'A') &
                #    (factor_df['securitytype'] == 'EQTY') &
                #    (factor_df['sharetype'] == 'NS')
                #]
                
                
                # Add size and liquidity filters
                factor_df = prev_data_df[
                    
                    # Size filter: market cap > $500M
                    (prev_data_df['dlycap'] > 500) &  # Cap in millions
                    # Price filter: price > $5
                    (prev_data_df['dlyprc'].abs() > 5)
                ]

                # Calculate factors and sort using previous day's data
                self.cal_factor_func(factor_df)
                factor_df.sort_values(by='point1', ascending=False, inplace=True)

                # Calculate new group assignments
                total_stocks = len(factor_df)
                for group_i in range(self.group_num):
                    start_idx = int(total_stocks * group_i / self.group_num)
                    end_idx = int(total_stocks * (group_i + 1) / self.group_num)
                    new_group_stocks = factor_df.index[start_idx:end_idx].tolist()
                    
                    # Calculate turnover
                    old_position = set(self.group_position_list[group_i])
                    new_position = set(new_group_stocks)
                    turnover = 1 - len(old_position & new_position) / len(old_position) if old_position else 0
                    self.daily_turnover_list[group_i].append(turnover)
                    
                    # Update positions
                    self.group_position_list[group_i] = new_group_stocks

    '''
    def backtest(self):
        """Run backtesting analysis."""
        # Initialize portfolio on first date
        start_date = self.trading_date[1]
        self.initialize_portfolio(start_date)
        
        # Process each date
        for date in tqdm(self.trading_date, desc='Processing dates'):
            current_data_df = self.data_df_dic[date]
            
            # Calculate daily returns
            self.calculate_daily_returns(date, current_data_df)
            
            # Rebalance if needed (excluding first date)
            if date in self.balance_date and date != start_date:
                self.rebalance_portfolio(date)
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
    
    '''
    def balance_func(self, date, current_data_df):
        """
        Perform portfolio rebalancing and daily return calculations.
        Calculate returns every day but only rebalance portfolios on balance dates.
        """
        # Calculate daily returns for current groups regardless of rebalance
        if len(self.group_position_list[0]) > 0:  # If we have existing positions
            # Get returns for all current positions
            day_returns = current_data_df[
                ['dlyret', 'dlyretmissflg', 'dlyclose', 'dlyopen']
            ]
            
            # Process returns for the day
            processed_returns = self.process_daily_returns(day_returns)
            
            # Calculate returns for each group
            for group_i in range(self.group_num):
                group_stocks = self.group_position_list[group_i]
                if len(group_stocks) > 0:
                    group_return = processed_returns.loc[
                        processed_returns.index.intersection(group_stocks), 
                        'dlyret'
                    ].mean()
                    self.group_daily_return_list[group_i].append(
                        group_return if not np.isnan(group_return) else 0
                    )
                else:
                    self.group_daily_return_list[group_i].append(0)
        
        # Rebalance portfolios only on balance dates
        if date in self.balance_date:
            # Get investable universe 
            factor_df = current_data_df.dropna(subset=['tradingstatusflg'])
            factor_df = factor_df[
                (factor_df['tradingstatusflg'] == 'A') &
                (factor_df['securitytype'] == 'EQTY') &
                (factor_df['sharetype'] == 'NS')
            ][self.factor_list].dropna()

            # Calculate factors and sort
            self.cal_factor_func(factor_df)
            factor_df.sort_values(by='point1', ascending=False, inplace=True)

            # Calculate new group assignments
            total_stocks = len(factor_df)
            for group_i in range(self.group_num):
                start_idx = int(total_stocks * group_i / self.group_num)
                end_idx = int(total_stocks * (group_i + 1) / self.group_num)
                new_group_stocks = factor_df.index[start_idx:end_idx].tolist()
                
                # Calculate turnover
                old_position = set(self.group_position_list[group_i])
                new_position = set(new_group_stocks)
                turnover = 1 - len(old_position & new_position) / len(old_position) if old_position else 0
                self.daily_turnover_list[group_i].append(turnover)
                
                # Update positions
                self.group_position_list[group_i] = new_group_stocks

    def backtest(self):
        """Run backtesting analysis."""
        for date in tqdm(self.trading_date, desc='Processing dates'):
            current_data_df = self.data_df_dic[date]
            self.balance_func(date, current_data_df)
        
        # Calculate cumulative returns
        self.cum_list = [
            np.cumprod(np.array(daily_returns) + 1).tolist()
            for daily_returns in self.group_daily_return_list
        ]
        
        # Calculate performance metrics
        trading_days_per_year = 252
        self.annual_returns = [
            (np.array(cum[-1]) ** (trading_days_per_year/len(cum)) - 1) * 100
            for cum in self.cum_list
        ]
        
        self.sharpe_ratios = [
            (np.mean(returns) * np.sqrt(trading_days_per_year)) / 
            (np.std(returns) * np.sqrt(trading_days_per_year))
            for returns in self.group_daily_return_list
        ]
        
        self.max_drawdowns = [
            np.max((np.maximum.accumulate(cum) - cum) / 
                   np.maximum.accumulate(cum)) * 100
            for cum in self.cum_list
        ]
    '''
    def plot_cumulative_returns(self, ax=None):
        """
        Plot cumulative returns with actual dates on x-axis.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, cum in enumerate(self.cum_list):
            label = f'Group {i+1} (Ann. Ret: {self.annual_returns[i]:.1f}%, SR: {self.sharpe_ratios[i]:.2f})'
            ax.plot(self.trading_date, cum, label=label, linewidth=2 if i in [0, -1] else 1)
        
        ax.set_title('Cumulative Returns by Group')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit number of x ticks
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        return ax

    def plot_turnover(self, ax=None):
        """Plot turnover for long and short portfolios."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        rebalance_dates = range(len(self.daily_turnover_list[0]))
        ax.plot(rebalance_dates, self.daily_turnover_list[0], 
                label='Long Portfolio', color='green')
        ax.plot(rebalance_dates, self.daily_turnover_list[-1], 
                label='Short Portfolio', color='red')
        
        ax.set_title('Portfolio Turnover')
        ax.set_xlabel('Rebalance Periods')
        ax.set_ylabel('Turnover Rate')
        ax.legend()
        ax.grid(True)
        return ax

    def plot_performance_summary(self):
        """Create comprehensive performance summary plot."""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_cumulative_returns(ax1)
        
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_turnover(ax2)
        
        ax3 = fig.add_subplot(gs[1, 1])
        metrics = pd.DataFrame({
            'Annual Return (%)': self.annual_returns,
            'Sharpe Ratio': self.sharpe_ratios,
            'Max Drawdown (%)': self.max_drawdowns
        }, index=[f'Group {i+1}' for i in range(self.group_num)])
        metrics.plot(kind='bar', ax=ax3)
        ax3.set_title('Performance Metrics by Group')
        ax3.grid(True)
        
        plt.tight_layout()
        return fig