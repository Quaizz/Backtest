import os
import numpy as np
import scipy.stats 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import warnings
from pathlib import Path
import mplcursors
import matplotlib
import calendar


class USQuantileFactorAnalysis:
    def __init__(self, data_df_dic,  factor_list, universe, 
                 group_num, trading_date, rebalance_freq, balance_day, ic_periods, gvkeyx='000003',factor_ascending=False, 
                 min_price=2.0, market_cap_percentile=0.5):
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
        factor_ascending : bool
        If True, lower factor values are better (Group 1 will have lowest values)
        If False, higher factor values are better (Group 1 will have highest values)
        """
        self.data_df_dic = data_df_dic
        self.factor_list = factor_list
        self.universe = universe
        self.group_num = group_num
        self.trading_date = trading_date
        self.rebalance_freq = rebalance_freq
        self.ic_periods = ic_periods
        self.balance_day = balance_day
        self.gvkeyx = gvkeyx

        # Calculate rebalancing dates first
        self.rebalance_date = self.check_trade_day(trading_date, rebalance_freq, balance_day)
        

        # Initialize storage for analysis results
        self.group_position_list = [{} for _ in range(group_num)]
        self.group_daily_return_list = [[] for _ in range(group_num)]
        
        # Performance metrics storage
        self.group_return_std_list = [[] for _ in range(group_num)]
        self.cum_list = []
        self.daily_turnover_list = [[] for _ in range(group_num)]
        

        # Initialize benchmark data members
        self.benchmark_returns = None
        self.benchmark_cum = None
        self.benchmark_ann_return = None

        self.universe_returns = []
        self.universe_cum = None

        self.factor_ascending = factor_ascending

        self.min_price = min_price
        self.market_cap_percentile = market_cap_percentile


        # Initialize IC lists
        self.ic_list = []
        self.rankic_list = []

        # Results storage
        self.annual_returns = None
        self.sharpe_ratios = None
        self.max_drawdowns = None
        self.daily_rf_rates = None
        self.size_dic = {}
        # Load benchmark data
        self._load_benchmark_data()

    def _load_benchmark_data(self):
        """
        Load benchmark data and initialize benchmark-related class members.
        """
        benchmark_file = Path(f'Data_all/Benchmark_data/{self.gvkeyx}/benchmark_{self.gvkeyx}.parquet')
        if not benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
            
        # Read benchmark data
        benchmark_df = pd.read_parquet(benchmark_file)
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        benchmark_df.set_index('date', inplace=True)
        
        # Get benchmark returns and prices for our trading dates
        trading_dates = pd.to_datetime(self.trading_date)[1:]
        self.benchmark_returns = benchmark_df.loc[trading_dates, 'ret_idx']
        self.benchmark_prices = benchmark_df.loc[trading_dates, 'tr_idx']
        
        # Calculate cumulative returns relative to first date's price
        base_price = self.benchmark_prices.iloc[0]  # Price at first date (previous to start date)
        self.benchmark_cum = self.benchmark_prices / base_price

        # Load risk-free rates
        rf_file = Path('Data_all/RF_data/rf_rates.parquet')
        if not rf_file.exists():
            raise FileNotFoundError("Risk-free rate file not found")
        
        rf_df = pd.read_parquet(rf_file)
        rf_df['date'] = pd.to_datetime(rf_df['date'])
        rf_df.set_index('date', inplace=True)
        
        # Get risk-free rates for our trading dates
        self.daily_rf_rates = rf_df.loc[trading_dates, 'rf']

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
            rerebalance_dates = dates_df[rebalance_mask]['date_str'].tolist()
            
        elif rebalance_freq == '1m':
            # Group dates by year-month
            dates_df['ym'] = dates_df['date'].dt.strftime('%Y%m')
            monthly_groups = dates_df.groupby('ym')
            
            rerebalance_dates = []
            for _, group in monthly_groups:
                group = group.sort_values('date')
                
                # Get the last possible trading day for this target day
                target_day = min(balance_day, group['date'].iloc[-1].day)
                valid_dates = group[group['date'].dt.day <= target_day]
                
                if not valid_dates.empty:
                    # Get the last valid trading day before or on target day
                    rerebalance_dates.append(valid_dates.iloc[-1]['date_str'])
                else:
                    # If no valid dates before target day, use month's first trading day
                    rerebalance_dates.append(group.iloc[0]['date_str'])
                    
        else:
            raise ValueError(f"Invalid rebalance frequency: {rebalance_freq}")
            
        # Filter out dates before start date and sort
        start_date = pd.to_datetime(first_date)
        rerebalance_dates = [date for date in rerebalance_dates 
                        if pd.to_datetime(date) >= start_date]

        # Ensure first trading date is included and dates are properly ordered
        if first_date not in rerebalance_dates:
            rerebalance_dates = [first_date] + sorted(rerebalance_dates)
        else:
            rerebalance_dates = sorted(rerebalance_dates)
            
        return rerebalance_dates
        
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

        # Calculate market cap percentile threshold (e.g., bottom 5%)
        #mktcap_threshold = prev_data_df['dlycap'].quantile(self.market_cap_percentile)
        # Apply filters for investable universe
        factor_df = prev_data_df[
            (prev_data_df['dlycap'] > prev_data_df['dlycap'].quantile(self.market_cap_percentile)) &  # Market cap > $500M
            (prev_data_df['dlyprc'] > self.min_price)      # Price > $5
        ]
        
        factor_df.dropna(subset=self.factor_list)

        # Calculate initial factor values
        self.cal_factor_func(factor_df)
        sort_ascending = self.factor_ascending
        factor_df.sort_values(by='point1', ascending=sort_ascending, inplace=True)
        
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
        """Calculate composite factor score from multiple factors"""
        # Rank each factor
        ranks = pd.DataFrame()
        for factor in self.factor_list:
            ranks[f"{factor}_rank"] = factor_df[factor].rank(ascending=self.factor_ascending)
        
        # Calculate composite score (average of ranks)
        factor_df['point1'] = ranks.sum(axis=1)

    def calculate_daily_returns(self, date, current_data_df):
        """Calculate daily returns for existing positions."""
        if len(self.group_position_list[0]) > 0:
            
            
            day_returns = current_data_df[
                ['dlyret', 'dlyretmissflg', 'dlyclose', 'dlyopen']
            ]
            
            processed_returns = self.process_daily_returns(day_returns)
            
            group_returns = []
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
                        group_returns.append(float(group_return) if not np.isnan(group_return) else 0)
                    else:
                        self.group_daily_return_list[group_i].append(0)
                        group_returns.append(0)
                else:
                    self.group_daily_return_list[group_i].append(0)
                    group_returns.append(0)
            self.universe_returns.append(np.mean(group_returns))
        else:
            self.universe_returns.append(0)
            for group_i in range(self.group_num):
                self.group_daily_return_list[group_i].append(0)
                
    def rebalance_portfolio(self, date):
        """Rebalance portfolio using previous day's data."""
        prev_date_idx = self.trading_date.index(date) - 1
        prev_date = self.trading_date[prev_date_idx]
        prev_data_df = self.data_df_dic[prev_date]
        
        # Calculate market cap percentile threshold (e.g., bottom 5%)
        
        # Apply filters for investable universe
        

        factor_df = prev_data_df[
            (prev_data_df['dlycap'] > prev_data_df['dlycap'].quantile(self.market_cap_percentile)) &  # Market cap > $500M
            (prev_data_df['dlyprc'] > self.min_price)      # Price > $5
        ]
        
        factor_df.dropna(subset=self.factor_list[0])

        # Calculate factors and sort
        self.cal_factor_func(factor_df)
        sort_ascending = self.factor_ascending
        factor_df.sort_values(by='point1', ascending=sort_ascending, inplace=True)
        
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

        # Calculate IC metrics
        self.calculate_ic(factor_df, date)

    def calculate_ic(self, factor_df, date):
        """
        Calculate IC metrics for given factor data and date.
        """
        curr_idx = self.trading_date.index(date)
        if curr_idx + self.ic_periods < len(self.trading_date):
            # Get all dates needed
            dates = self.trading_date[curr_idx:curr_idx + self.ic_periods + 1]
            
            # Initialize returns matrix
            stocks = factor_df.index
            returns_matrix = np.zeros((len(stocks), len(dates)))
            
            # Process all dates at once
            for i, day_date in enumerate(dates):
                day_data = self.data_df_dic[day_date]
                day_data = self.process_daily_returns(
                    day_data.loc[:, ['dlyret', 'dlyretmissflg', 'dlyclose', 'dlyopen']]
                )
                
                # Get returns for our stocks
                common_stocks = stocks.intersection(day_data.index)
                if len(common_stocks) > 0:
                    returns_matrix[
                        [stocks.get_loc(s) for s in common_stocks], i
                    ] = day_data.loc[common_stocks, 'dlyret'].values
            
            # Calculate compound returns
            compound_returns = np.prod(returns_matrix + 1, axis=1) - 1
            factor_df['rate'] = compound_returns
            factor_df.dropna(subset=['rate', 'point1'],inplace=True)

            # Calculate ICs
            if len(factor_df) > 0:
                # Regular IC
                ic = scipy.stats.pearsonr(factor_df['rate'].values, factor_df['point1'].values)[0]
                self.ic_list.append(ic if not np.isnan(ic) else 0)
                
                # Rank IC (Spearman)
                rank_ic = scipy.stats.spearmanr(factor_df['rate'].values, factor_df['point1'].values)[0]
                self.rankic_list.append(rank_ic if not np.isnan(rank_ic) else 0)
            else:
                self.ic_list.append(0)
                self.rankic_list.append(0)


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
            # Calculate excess returns over rf
            excess_returns = returns_array - self.daily_rf_rates.values
            # Annualize mean and standard deviation
            mean_excess_return = np.mean(excess_returns) * trading_days_per_year
            std_return = np.std(excess_returns) * np.sqrt(trading_days_per_year)
            sharpe = mean_excess_return / std_return if std_return != 0 else 0
            self.sharpe_ratios.append(sharpe)
        
        # Calculate benchmark metrics
        if hasattr(self, 'benchmark_returns'):
            benchmark_excess = self.benchmark_returns - self.daily_rf_rates.values
            self.benchmark_sharpe = (np.mean(benchmark_excess) * trading_days_per_year) / \
                                (np.std(benchmark_excess) * np.sqrt(trading_days_per_year))
            # Calculate benchmark annual return
            total_benchmark_return = (1 + np.array(self.benchmark_returns)).prod() - 1
            years = len(self.benchmark_returns) / trading_days_per_year
            self.benchmark_ann_return = ((1 + total_benchmark_return) ** (1/years) - 1) * 100

            
        # Calculate universe cumulative returns
        self.universe_cum = np.cumprod(1 + np.array(self.universe_returns)).tolist()
        
        # Calculate universe annual return
        total_universe_return = (1 + np.array(self.universe_returns)).prod() - 1
        years = len(self.universe_returns) / trading_days_per_year
        self.universe_ann_return = ((1 + total_universe_return) ** (1/years) - 1) * 100
        
        # Calculate universe Sharpe ratio
        universe_excess = np.array(self.universe_returns) - self.daily_rf_rates.values
        self.universe_sharpe = (np.mean(universe_excess) * trading_days_per_year) / \
                            (np.std(universe_excess) * np.sqrt(trading_days_per_year))
        # Calculate maximum drawdowns
        self.max_drawdowns = []
        for cum_rets in self.cum_list:
            cum_returns = np.array(cum_rets)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (running_max - cum_returns) / running_max
            max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
            self.max_drawdowns.append(max_drawdown)


    def backtest(self):
        """Run backtesting analysis."""
        # Initialize portfolio on first date
        start_date = self.trading_date[1]
        self.initialize_portfolio(start_date)
        
        # Process each date
        for date in tqdm(self.trading_date[1:], desc='Processing dates'):
            current_data_df = self.data_df_dic[date]
            
            # Calculate daily returns
            self.calculate_daily_returns(date, current_data_df)
            
            # Rebalance if needed (excluding first date)
            if date in self.rebalance_date and date != start_date:
                self.rebalance_portfolio(date)
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
    
    def plot_cumulative_returns(self, ax=None, show_benchmark=True):
        """
        Basic non-interactive version of cumulative returns plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        if show_benchmark and hasattr(self, 'benchmark_cum'):
            # Calculate benchmark annualized return
            total_benchmark_return = (1 + np.array(self.benchmark_returns)).prod() - 1
            years = len(self.benchmark_returns) / 252
            benchmark_ann_return = ((1 + total_benchmark_return) ** (1/years) - 1) * 100
            
            # Use pre-calculated benchmark Sharpe ratio
            benchmark_label = f'Benchmark (Ann. Ret: {benchmark_ann_return:.1f}%, SR: {self.benchmark_sharpe:.2f})'
            ax.plot(self.trading_date[1:], self.benchmark_cum,
                label=benchmark_label, color='black', linestyle='--', linewidth=1)
        
        # Add universe line
        universe_label = f'Universe (Ann. Ret: {self.universe_ann_return:.1f}%, SR: {self.universe_sharpe:.2f})'
        ax.plot(self.trading_date[1:], self.universe_cum,
                label=universe_label, color='gray', linestyle=':', linewidth=1)
        
        for i, cum in enumerate(self.cum_list):
            label = f'Group {i+1} (Ann. Ret: {self.annual_returns[i]:.1f}%, SR: {self.sharpe_ratios[i]:.2f})'
            ax.plot(self.trading_date[1:], cum, label=label, linewidth=2 if i in [0, -1] else 1)
        
        ax.set_title('Cumulative Returns by Group')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        return ax

    def plot_excess_returns(self, ax=None):
        """
        Basic non-interactive version of excess returns plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        total_benchmark_return = (1 + np.array(self.benchmark_returns)).prod() - 1
        years = len(self.benchmark_returns) / 252
        benchmark_ann_return = ((1 + total_benchmark_return) ** (1/years) - 1) * 100
        
        for i, cum in enumerate(self.cum_list):
            excess_cum = np.array(cum) - np.array(self.benchmark_cum)
            excess_ann_return = self.annual_returns[i] - benchmark_ann_return
            
            excess_daily_returns = np.array(self.group_daily_return_list[i]) - self.benchmark_returns
            excess_sharpe = (np.mean(excess_daily_returns) * 252) / (np.std(excess_daily_returns) * np.sqrt(252))
            
            label = f'Group {i+1} (Excess Ann. Ret: {excess_ann_return:.1f}%, IR: {excess_sharpe:.2f})'
            ax.plot(self.trading_date[1:], excess_cum,
                label=label, linewidth=2 if i in [0, -1] else 1)
        
        ax.set_title('Cumulative Excess Returns over Benchmark')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Excess Return')
        
        ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        return ax
    
    def plot_cumulative_returns_interactive(self, ax=None, show_benchmark=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.figure
        
        lines = []  # Store line objects for interactivity
        
        # Plot universe line first
        universe_label = f'Universe (Ann. Ret: {self.universe_ann_return:.1f}%, SR: {self.universe_sharpe:.2f})'
        line = ax.plot(self.trading_date[1:], self.universe_cum,
                    label=universe_label, color='gray', linestyle=':', linewidth=1)[0]
        lines.append(line)
        
        # Calculate and plot benchmark if requested
        if show_benchmark and hasattr(self, 'benchmark_cum'):
            total_benchmark_return = (1 + np.array(self.benchmark_returns)).prod() - 1
            years = len(self.benchmark_returns) / 252
            benchmark_ann_return = ((1 + total_benchmark_return) ** (1/years) - 1) * 100
            
            benchmark_label = f'Benchmark (Ann. Ret: {benchmark_ann_return:.1f}%, SR: {self.benchmark_sharpe:.2f})'
            line = ax.plot(self.trading_date[1:], self.benchmark_cum,
                        label=benchmark_label, color='black', linestyle='--', linewidth=1)[0]
            lines.append(line)
        
        # Plot group returns
        for i, cum in enumerate(self.cum_list):
            label = f'Group {i+1} (Ann. Ret: {self.annual_returns[i]:.1f}%, SR: {self.sharpe_ratios[i]:.2f})'
            line = ax.plot(self.trading_date[1:], cum, label=label, linewidth=2 if i in [0, -1] else 1)[0]
            lines.append(line)
        
        ax.set_title('Cumulative Returns by Group')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # Create legend with clickable behavior
        leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        lined = {}
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)  # Set picker to a scalar value
            lined[legline] = origline

        def on_pick(event):
            # Only handle legend line pick events
            if event.artist in lined:
                legline = event.artist
                origline = lined[legline]
                visible = not origline.get_visible()
                origline.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
        
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        # Add hover annotations
        cursor = mplcursors.cursor(lines, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.target
            date_idx = np.abs(matplotlib.dates.date2num(self.trading_date) - x).argmin()
            line_idx = lines.index(sel.artist)
            
            rf_rate = self.daily_rf_rates.iloc[date_idx] * 100
            
            if line_idx == 0:  # Universe
                daily_ret = self.universe_returns[date_idx] * 100
                cum_ret = (self.universe_cum[date_idx] - 1) * 100
                excess_ret = daily_ret - rf_rate
                label = "Universe"
            elif line_idx == 1 and show_benchmark:  # Benchmark
                daily_ret = self.benchmark_returns[date_idx] * 100
                cum_ret = (self.benchmark_cum[date_idx] - 1) * 100
                excess_ret = daily_ret - rf_rate
                label = "Benchmark"
            else:  # Groups
                group_idx = line_idx - (2 if show_benchmark else 1)  # Adjust for universe and benchmark
                daily_ret = self.group_daily_return_list[group_idx][date_idx] * 100
                cum_ret = (self.cum_list[group_idx][date_idx] - 1) * 100
                excess_ret = daily_ret - rf_rate
                label = f"Group {group_idx + 1}"
            
            date_str = self.trading_date[date_idx].strftime('%Y-%m-%d')
            sel.annotation.set_text(
                f"{label}\nDate: {date_str}\n"
                f"Daily Return: {daily_ret:.2f}%\n"
                #f"RF Rate: {rf_rate:.3f}%\n"
                #f"Excess Return: {excess_ret:.2f}%\n"
                f"Cumulative Return: {cum_ret:.2f}%"
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        
        ax.grid(True)
        return ax

    def plot_excess_returns_interactive(self, ax=None):
        """
        Plot interactive excess returns with clickable legend and hover information.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.figure
        
        # Calculate benchmark metrics
        total_benchmark_return = (1 + np.array(self.benchmark_returns)).prod() - 1
        years = len(self.benchmark_returns) / 252
        benchmark_ann_return = ((1 + total_benchmark_return) ** (1/years) - 1) * 100
        self.benchmark_ann_return = benchmark_ann_return
        lines = []  # Store line objects for interactivity
        
        # Calculate and plot excess returns
        for i, cum in enumerate(self.cum_list):
            excess_cum = np.array(cum) - np.array(self.benchmark_cum)
            excess_ann_return = self.annual_returns[i] - benchmark_ann_return
            
            excess_daily_returns = np.array(self.group_daily_return_list[i]) - self.benchmark_returns
            excess_sharpe = (np.mean(excess_daily_returns) * 252) / (np.std(excess_daily_returns) * np.sqrt(252))
            
            label = f'Group {i+1} (Excess Ann. Ret: {excess_ann_return:.1f}%, IR: {excess_sharpe:.2f})'
            line = ax.plot(self.trading_date[1:], excess_cum,
                        label=label, linewidth=2 if i in [0, -1] else 1)[0]
            lines.append(line)
        
        ax.set_title('Cumulative Excess Returns over Benchmark')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Excess Return')
        
 
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # Create legend with clickable behavior
        leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        lined = {}
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)  # Set picker to a scalar value
            lined[legline] = origline

        def on_pick(event):
            # Only handle legend line pick events
            if event.artist in lined:
                legline = event.artist
                origline = lined[legline]
                visible = not origline.get_visible()
                origline.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
        
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        # Add hover annotations
        cursor = mplcursors.cursor(lines, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.target
            date_idx = np.abs(matplotlib.dates.date2num(self.trading_date) - x).argmin()
            line_idx = lines.index(sel.artist)
            
            # Calculate daily and cumulative excess returns
            daily_group_ret = self.group_daily_return_list[line_idx][date_idx] * 100
            daily_bench_ret = self.benchmark_returns[date_idx] * 100
            daily_excess = daily_group_ret - daily_bench_ret
            cum_excess = y * 100  # Already in excess return form
            
            rf_rate = self.daily_rf_rates.iloc[date_idx] * 100
            
            date_str = self.trading_date[date_idx].strftime('%Y-%m-%d')
            sel.annotation.set_text(
                f"Group {line_idx + 1}\nDate: {date_str}\n"
                f"Group Return: {daily_group_ret:.2f}%\n"
                f"Benchmark Return: {daily_bench_ret:.2f}%\n"
                #f"RF Rate: {rf_rate:.3f}%\n"
                f"Daily Excess: {daily_excess:.2f}%\n"
                f"Cumulative Excess: {cum_excess:.2f}%"
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        
        ax.grid(True)
        return ax

    def plot_universe_excess_returns(self, ax=None):
        """
        Basic non-interactive version of excess returns over universe plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get universe metrics for labels
        total_universe_return = (1 + np.array(self.universe_returns)).prod() - 1
        years = len(self.universe_returns) / 252
        universe_ann_return = ((1 + total_universe_return) ** (1/years) - 1) * 100
        
        for i, cum in enumerate(self.cum_list):
            excess_cum = np.array(cum) - np.array(self.universe_cum)
            excess_ann_return = self.annual_returns[i] - self.universe_ann_return
            
            # Calculate excess Sharpe over universe
            excess_daily_returns = np.array(self.group_daily_return_list[i]) - self.universe_returns
            excess_sharpe = (np.mean(excess_daily_returns) * 252) / (np.std(excess_daily_returns) * np.sqrt(252))
            
            label = f'Group {i+1} (Excess Ann. Ret: {excess_ann_return:.1f}%, IR: {excess_sharpe:.2f})'
            ax.plot(self.trading_date[1:], excess_cum,
                label=label, linewidth=2 if i in [0, -1] else 1)
        
        ax.set_title('Cumulative Excess Returns over Universe')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Excess Return')
        
        ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        return ax

    def plot_universe_excess_returns_interactive(self, ax=None):
        """
        Interactive version of excess returns over universe plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.figure
        
        lines = []  # Store line objects for interactivity
        
        # Calculate universe metrics
        total_universe_return = (1 + np.array(self.universe_returns)).prod() - 1
        years = len(self.universe_returns) / 252
        universe_ann_return = ((1 + total_universe_return) ** (1/years) - 1) * 100
        
        # Calculate and plot excess returns
        for i, cum in enumerate(self.cum_list):
            excess_cum = np.array(cum) - np.array(self.universe_cum)
            excess_ann_return = self.annual_returns[i] - self.universe_ann_return
            
            excess_daily_returns = np.array(self.group_daily_return_list[i]) - self.universe_returns
            excess_sharpe = (np.mean(excess_daily_returns) * 252) / (np.std(excess_daily_returns) * np.sqrt(252))
            
            label = f'Group {i+1} (Excess Ann. Ret: {excess_ann_return:.1f}%, IR: {excess_sharpe:.2f})'
            line = ax.plot(self.trading_date[1:], excess_cum,
                        label=label, linewidth=2 if i in [0, -1] else 1)[0]
            lines.append(line)
        
        ax.set_title('Cumulative Excess Returns over Universe')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Excess Return')
        
        ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # Create legend with clickable behavior
        leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        lined = {}
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)
            lined[legline] = origline
        
        def on_pick(event):
            if event.artist in lined:
                legline = event.artist
                origline = lined[legline]
                visible = not origline.get_visible()
                origline.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
        
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        # Add hover annotations
        cursor = mplcursors.cursor(lines, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.target
            date_idx = np.abs(matplotlib.dates.date2num(self.trading_date) - x).argmin()
            line_idx = lines.index(sel.artist)
            
            # Calculate daily and cumulative excess returns
            daily_group_ret = self.group_daily_return_list[line_idx][date_idx] * 100
            daily_universe_ret = self.universe_returns[date_idx] * 100
            daily_excess = daily_group_ret - daily_universe_ret
            cum_excess = y * 100  # Already in excess return form
            
            date_str = self.trading_date[date_idx].strftime('%Y-%m-%d')
            sel.annotation.set_text(
                f"Group {line_idx + 1}\nDate: {date_str}\n"
                f"Group Return: {daily_group_ret:.2f}%\n"
                f"Universe Return: {daily_universe_ret:.2f}%\n"
                f"Daily Excess: {daily_excess:.2f}%\n"
                f"Cumulative Excess: {cum_excess:.2f}%"
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        
        ax.grid(True)
        return ax


    def plot_turnover(self, ax=None):
        """Plot turnover for long and short portfolios."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        rerebalance_dates = range(len(self.daily_turnover_list[0]))
        ax.plot(rerebalance_dates, self.daily_turnover_list[0], 
                label='Long Portfolio', color='green')
        ax.plot(rerebalance_dates, self.daily_turnover_list[-1], 
                label='Short Portfolio', color='red')
        
        ax.set_title('Portfolio Turnover')
        ax.set_xlabel('Rebalance Periods')
        ax.set_ylabel('Turnover Rate')
        ax.legend()
        ax.grid(True)
        return ax
    
    def plot_strategy_analysis(self, plot_type, ax=None, interactive=False):
        """
        Plot strategy analysis with drawdowns.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.figure
        
        # Create twin axis for drawdowns
        ax2 = ax.twinx()
        lines = []  # For interactive version
        
        # Prepare data based on plot type
        if plot_type == 'long_short':
            # Change from ratio to difference of cumulative returns
            strategy_returns = np.array(self.group_daily_return_list[0]) - np.array(self.group_daily_return_list[-1])
            ann_return = self.annual_returns[0] - self.annual_returns[-1]
            title = f'Long Best Short Worst Return {ann_return:.1f}%'
            
        elif plot_type == 'long_universe':
            # Change from ratio to difference with universe returns
            strategy_returns = np.array(self.group_daily_return_list[0]) - np.array(self.universe_returns)
            ann_return = self.annual_returns[0] - self.universe_ann_return
            title = f'Long Best-Universe Alpha {ann_return:.1f}%'
            
        elif plot_type == 'short_universe':
            # Change from ratio to difference with universe returns
            strategy_returns = np.array(self.group_daily_return_list[-1]) - np.array(self.universe_returns)
            ann_return = self.annual_returns[-1] - self.universe_ann_return
            title = f'Short Worst-Universe Alpha {ann_return:.1f}%'
            
        elif plot_type == 'long_benchmark':
            # Change from ratio to difference with benchmark returns
            strategy_returns = np.array(self.group_daily_return_list[0]) - np.array(self.benchmark_returns)
            ann_return = self.annual_returns[0] - self.benchmark_ann_return
            title = f'Long Best-Benchmark Alpha {ann_return:.1f}%'
        
        # Calculate drawdowns correctly
        cum_strategy = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cum_strategy)
        drawdowns = (cum_strategy - running_max) / running_max * 100
        max_drawdown = np.min(drawdowns)
        # Calculate cumulative returns for plotting
        cum_returns = cum_strategy - 1

        # Plot strategy returns with legend including annual return
        line = ax.plot(self.trading_date[1:], cum_returns * 100, 
                    color='blue', linewidth=1.5, 
                    label=f'Strategy (Ann. Ret: {ann_return:.1f}%, Max DD: {max_drawdown:.1f}%)')[0]
        if interactive:
            lines.append(line)
        
        # Plot drawdown line with legend
        draw_line = ax2.plot(self.trading_date[1:], drawdowns, 
                            color='red', alpha=0.7, linewidth=0.5,
                            label=f'Drawdown')[0]
        if interactive:
            lines.append(draw_line)
        
        # Mark maximum drawdown period and get dates
        max_dd_idx = np.argmin(drawdowns)
        peak_idx = np.where(drawdowns[:max_dd_idx] == 0)[0]
        max_dd_start_idx = peak_idx[-1] if len(peak_idx) > 0 else 0

        # Get the dates for max drawdown period
        max_dd_start_date = self.trading_date[1+max_dd_start_idx].strftime('%Y-%m-%d')
        max_dd_end_date = self.trading_date[1+max_dd_idx].strftime('%Y-%m-%d')

        max_dd_line = ax2.plot(self.trading_date[1+max_dd_start_idx:1+max_dd_idx+1], 
                            drawdowns[max_dd_start_idx:max_dd_idx+1], 
                            color='green', linewidth=1, alpha=0.8,
                            label=f'Max Drawdown Period ({max_dd_start_date} to {max_dd_end_date})')[0]
        if interactive:
            lines.append(max_dd_line)
        
        # Set labels and format axes
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')
        ax2.set_ylabel('Drawdown (%)')
        
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.grid(True)
        
        # Set y-axis limits dynamically
        max_ret = np.max(cum_returns) * 100
        min_ret = np.min(cum_returns) * 100
        padding = (max_ret - min_ret) * 0.1 
        ax.set_ylim(min_ret- padding, max_ret + padding)  # Add 10% padding
        
        # Invert right y-axis for drawdowns and set dynamic limits
        ax2.invert_yaxis()
        ax2.set_ylim(bottom=max(max_drawdown * 1.2, -100), top=0)  # Set bottom limit to 120% of max drawdown or -100%
        
        # Add legends for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if interactive:
            cursor = mplcursors.cursor(lines, hover=True)
            
            @cursor.connect("add")
            def on_hover(sel):
                x, y = sel.target
                date_idx = np.abs(matplotlib.dates.date2num(self.trading_date) - x).argmin()
                date_str = self.trading_date[date_idx].strftime('%Y-%m-%d')
                
                if sel.artist == lines[0]:  # Strategy line
                    sel.annotation.set_text(
                        f"Date: {date_str}\n"
                        f"Return: {y:.2f}%"
                    )
                elif sel.artist == lines[1]:  # Drawdown line
                    sel.annotation.set_text(
                        f"Date: {date_str}\n"
                        f"Drawdown: {y:.2f}%"
                    )
                else:  # Max drawdown period
                    sel.annotation.set_text(
                        f"Date: {date_str}\n"
                        f"Drawdown: {y:.2f}%\n"
                        f"Max Drawdown Period"
                    )
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        
        return ax, ax2


    def plot_all_strategies(self, interactive=False):
        """Plot all four strategy analysis plots."""
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
        
        plot_types = ['long_short', 'long_universe', 'short_universe', 'long_benchmark']
        
        for i, plot_type in enumerate(plot_types):
            ax = fig.add_subplot(gs[i])
            self.plot_strategy_analysis(plot_type, ax, interactive=interactive)
        
        plt.tight_layout()
        return fig
    
    def calculate_monthly_stats(self, strategy_type):
        """
        Calculate monthly statistics for a given strategy type.
        
        Parameters:
        -----------
        strategy_type : str
            One of: 'long_short', 'long_universe', 'short_universe', 'long_benchmark'
        
        Returns:
        --------
        DataFrame: Monthly and yearly statistics including returns, drawdowns, and win rates
        """
        # Get daily returns first
        if strategy_type == 'long_short':
            daily_returns = np.array(self.group_daily_return_list[0]) - np.array(self.group_daily_return_list[-1])
        elif strategy_type == 'long_universe':
            daily_returns = np.array(self.group_daily_return_list[0]) - np.array(self.universe_returns)
        elif strategy_type == 'short_universe':
            daily_returns = np.array(self.group_daily_return_list[-1]) - np.array(self.universe_returns)
        elif strategy_type == 'long_benchmark':
            daily_returns = np.array(self.group_daily_return_list[0]) - np.array(self.benchmark_returns)
        # Convert dates to pandas datetime
        dates = pd.to_datetime(self.trading_date[1:])
        
        # Create DataFrame with dates and daily returns
        df = pd.DataFrame({
            'date': pd.to_datetime(self.trading_date[1:]),
            'return': daily_returns
        })

        # Calculate monthly returns by compounding daily returns
        df_monthly = df.set_index('date')
        monthly_returns = []
        for name, group in df_monthly.groupby(pd.Grouper(freq='M')):
            if not group.empty:
                # Compound daily returns within month
                month_return = np.prod(1 + group['return']) - 1
                monthly_returns.append({'date': name, 'return': month_return})

        monthly_returns = pd.DataFrame(monthly_returns).set_index('date')['return']
        
        # Create yearly summary
        yearly_stats = []
        
        for year in monthly_returns.index.year.unique():
            year_data = monthly_returns[monthly_returns.index.year == year]
            
            # Calculate monthly returns for this year
            monthly_data = {}
            for month in range(1, 13):
                if month in year_data.index.month:
                    monthly_data[f'{calendar.month_abbr[month]}%'] = year_data[year_data.index.month == month].iloc[0] * 100
                else:
                    monthly_data[f'{calendar.month_abbr[month]}%'] = np.nan
            
            # Calculate yearly total return
            total_return = ((1 + year_data).prod() - 1) * 100
            
            # Calculate yearly max drawdown
            year_mask = df['date'].dt.year == year
            year_returns = df[year_mask]['return'].values
            
            cum_returns = np.cumprod(1 + year_returns) 
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max * 100
            max_dd = np.min(drawdowns)
            
            # Find max drawdown period
            max_dd_idx = np.argmin(drawdowns)
            peak_idx = np.where(drawdowns[:max_dd_idx] == 0)[0]
            max_dd_start_idx = peak_idx[-1] if len(peak_idx) > 0 else 0
            
            start_date = df[year_mask].iloc[max_dd_start_idx]['date'].strftime('%Y%m%d')
            end_date = df[year_mask].iloc[max_dd_idx]['date'].strftime('%Y%m%d')
            
            # Calculate win rate
            month_odds = (year_data > 0).mean() * 100
            
            # Combine all statistics
            year_stats = {
                'Year': year,
                **monthly_data,
                'Total%': total_return,
                'YMDD%': max_dd,
                'Start_date': start_date,
                'End_date': end_date,
                'Month_odds%': month_odds
            }
            
            yearly_stats.append(year_stats)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(yearly_stats)
        
        return result_df
    



    def plot_performance_summary(self, interactive=False):
        """Create comprehensive performance summary plot with benchmark and universe comparison."""
        # Create figure with increased height to accommodate tables
        num_years = len(self.trading_date) // 252  # Estimate number of years based on trading days (252 per year)
        base_height = 70  # Base height for plots
        table_height = max(3, num_years * 5)  # Adjust table height dynamically
        total_height = base_height + table_height  # Total height for figure

        # Create figure with dynamic height
        fig = plt.figure(figsize=(16, total_height))
        
        tableheight = num_years /8 
        # Create grid for plots, tables, and metrics (13 rows total)
        gs = fig.add_gridspec(15, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1, tableheight, tableheight, tableheight, tableheight, 0.35 ,1 ,1, 1])
        
        if interactive:
            # First 7 plots remain the same (0-6)
            ax1 = fig.add_subplot(gs[0, :])
            self.plot_cumulative_returns_interactive(ax1, show_benchmark=True)
            
            ax2 = fig.add_subplot(gs[1, :])
            self.plot_excess_returns_interactive(ax2)
            
            ax3 = fig.add_subplot(gs[2, :])
            self.plot_universe_excess_returns_interactive(ax3)
            
            plot_types = ['long_short', 'long_universe', 'short_universe', 'long_benchmark']
            for i, plot_type in enumerate(plot_types):
                ax = fig.add_subplot(gs[3+i, :])
                self.plot_strategy_analysis(plot_type, ax, interactive=True)
        else:
            ax1 = fig.add_subplot(gs[0, :])
            self.plot_cumulative_returns(ax1, show_benchmark=True)
            
            ax2 = fig.add_subplot(gs[1, :])
            self.plot_excess_returns(ax2)
            
            ax3 = fig.add_subplot(gs[2, :])
            self.plot_universe_excess_returns(ax3)
            
            plot_types = ['long_short', 'long_universe', 'short_universe', 'long_benchmark']
            for i, plot_type in enumerate(plot_types):
                ax = fig.add_subplot(gs[3+i, :])
                self.plot_strategy_analysis(plot_type, ax, interactive=False)
        
        # Add strategy statistics tables (rows 7-10)
        strategies = ['long_short', 'long_universe', 'short_universe', 'long_benchmark']
        for i, strategy in enumerate(strategies):
            ax_table = fig.add_subplot(gs[7+i, :])
            stats_df = self.calculate_monthly_stats(strategy)

            stats_df = stats_df.round(2)
            # Hide axis
            ax_table.axis('off')
            
            # Create table
            table = ax_table.table(
                cellText=stats_df.values,
                colLabels=stats_df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            ax_table.set_title(f"{strategy.replace('_', '-').title()} Monthly Statistics",
                            pad=20, fontsize=12, fontweight='bold')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            # Set different column widths
            month_cols = ['Jan%', 'Feb%', 'Mar%', 'Apr%', 'May%', 'Jun%', 
                        'Jul%', 'Aug%', 'Sep%', 'Oct%', 'Nov%', 'Dec%']
            for (row, col), cell in table._cells.items():
                if col == 0:  # Year column
                    cell.set_width(0.05)  # Year
                elif stats_df.columns[col] in month_cols:
                    cell.set_width(0.04)  # Monthly returns (narrower)
                else:
                    cell.set_width(0.08)  # Other columns (wider)
            table.scale(2.5, 1.2)
            
            # Color header and alternating rows
            for i in range(len(stats_df.index) + 1):  # +1 for header row
                for j in range(len(stats_df.columns)):
                    cell = table._cells.get((i, j))
                    if cell is not None:  # Check if cell exists
                        if i == 0:  # Header row
                            cell.set_facecolor('#ADD8E6')  # Light blue
                        elif i % 2:  # Alternating rows
                            cell.set_facecolor('#F0F0F0')  # Light gray

        # Add IC stats table
        ax_ic_stats = fig.add_subplot(gs[11,:])
        ic_stats = self.calculate_ic_stats()
        if ic_stats is not None:
            ax_ic_stats.axis('off')
            table = ax_ic_stats.table(
                cellText=ic_stats.values,
                colLabels=ic_stats.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(2.5, 1.2)
            
            # Style the table
            for (row, col), cell in table._cells.items():
                if row == 0:  # Header
                    cell.set_facecolor('#ADD8E6')
                cell.set_height(0.2)
            
            ax_ic_stats.set_title('IC Statistics', pad=20)
        

        # After the IC stats table and before the cumulative IC plot, add:
        ax_rankic_series = fig.add_subplot(gs[12,:])
        if interactive:
            self.plot_rankic_series_interactive(ax_rankic_series)
        else:
            self.plot_rankic_series(ax_rankic_series)

        ax_ic = fig.add_subplot(gs[13,:])
        if interactive:
            # ... other interactive plots ...
            self.plot_cumulative_ic_interactive(ax_ic)
        else:
            # Add cumulative IC plot
            self.plot_cumulative_ic(ax_ic)
        
        # Add turnover and metrics plots at the bottom (row 12)
        ax_turnover = fig.add_subplot(gs[14, 0])
        self.plot_turnover(ax_turnover)
        
        ax_metrics = fig.add_subplot(gs[14, 1])
        metrics = pd.DataFrame({
            'Annual Return (%)': self.annual_returns,
            'Sharpe Ratio': self.sharpe_ratios,
            'Max Drawdown (%)': self.max_drawdowns
        }, index=[f'Group {i+1}' for i in range(self.group_num)])
        metrics.plot(kind='bar', ax=ax_metrics)
        ax_metrics.set_title('Performance Metrics by Group')
        ax_metrics.grid(True)
        
        # Adjust layout
        #fig.suptitle('Factor Performance Analysis', fontsize=16, y=0.95)
        plt.tight_layout()
        return fig
    

    def calculate_ic_stats(self):
        """Calculate IC statistics for display."""
        if len(self.rankic_list) != 0:
            stats = {
                'IC%': round(np.nanmean(self.ic_list) * 100, 2),
                'RankIC%': round(np.nanmean(self.rankic_list) * 100, 2),
                'Annual RankICIR': round((np.nanmean(self.rankic_list) / np.nanstd(self.rankic_list)) * 
                                    (12**0.5 if self.ic_periods == 20 else
                                        52**0.5 if self.ic_periods == 5 else
                                        250**0.5 if self.ic_periods == 1 else 0), 2),
                '|RankIC|>0.02': round(100 * len([ic for ic in self.rankic_list if abs(ic) > 0.02]) / 
                                    len(self.rankic_list), 2),
                'P_Value': round(scipy.stats.ttest_1samp(self.rankic_list, 0)[1], 2),
                'RankIC Reverse': round(100 * len([ic for ic in self.rankic_list if 
                                                (ic > 0 if np.nanmean(self.rankic_list) <= 0 else ic < 0)]) / 
                                    len(self.rankic_list), 2),
                'RankIC Skew': round(scipy.stats.skew(self.rankic_list), 2),
                'RankIC Kurtosis': round(scipy.stats.kurtosis(self.rankic_list), 2)
            }
            return pd.DataFrame([stats])
        return None
    
    def plot_rankic_series(self, ax=None):
        """Plot rank IC time series."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate mean rank IC
        mean_rankic = np.mean(self.rankic_list)
        pct_positive = np.mean(np.array(self.rankic_list) > 0) * 100
        # Plot rank IC values
        ax.plot(self.rebalance_date[1:len(self.rankic_list)+1], self.rankic_list, 
                color='blue', linewidth=1)
        
        # Add horizontal lines at 0, 0.02, and -0.02
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.02, color='gray', linestyle='--', linewidth=0.5)
        ax.axhline(y=-0.02, color='gray', linestyle='--', linewidth=0.5)
        

        # Add mean rank IC line
        ax.axhline(y=mean_rankic, color='red', linestyle='-', linewidth=1, 
               label=f'Mean Rank IC: {mean_rankic:.3f}')
    
        ax.set_title('Rank IC Time Series')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rank IC')
        ax.grid(True)


        # Create legend with more information
        # Create legend with more information
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', label=f'Rank IC (Mean: {mean_rankic:.3f}, {pct_positive:.1f}% Positive)'),
            Line2D([0], [0], color='red', label=f'Mean = {mean_rankic:.3f}'),
            Line2D([0], [0], color='gray', linestyle='--', label='±0.02 Bounds'),
            Line2D([0], [0], color='black', label='Zero')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        

        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        return ax

    def plot_rankic_series_interactive(self, ax=None):
        """Plot interactive rank IC time series."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        # Calculate mean rank IC
        mean_rankic = np.mean(self.rankic_list)  
        pct_positive = np.mean(np.array(self.rankic_list) > 0) * 100

        # Plot rank IC values
        line = ax.plot(self.rebalance_date[1:len(self.rankic_list)+1], self.rankic_list, 
                    color='blue', linewidth=1)[0]
        
        # Add horizontal lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.02, color='gray', linestyle='--', linewidth=0.5)
        ax.axhline(y=-0.02, color='gray', linestyle='--', linewidth=0.5)
        # Add mean rank IC line
        mean_line = ax.axhline(y=mean_rankic, color='red', linestyle='-', linewidth=1,
                          label=f'Mean Rank IC: {mean_rankic:.3f}')
        ax.set_title('Rank IC Time Series')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rank IC')
        ax.grid(True)

        # Create legend with more information
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', label=f'Rank IC (Mean: {mean_rankic:.3f}, {pct_positive:.1f}% Positive)'),
            Line2D([0], [0], color='red', label=f'Mean = {mean_rankic:.3f}'),
            Line2D([0], [0], color='gray', linestyle='--', label='±0.02 Bounds'),
            Line2D([0], [0], color='black', label='Zero')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # Add hover annotations
        cursor = mplcursors.cursor([line], hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.target
            date_idx = np.abs(matplotlib.dates.date2num(self.rebalance_date[1:len(self.rankic_list)+1]) - x).argmin()
            date_str = self.rebalance_date[date_idx + 1].strftime('%Y-%m-%d')
            sel.annotation.set_text(
                f"Date: {date_str}\n"
                f"Rank IC: {y:.3f}"
                f"Mean Rank IC: {mean_rankic:.3f}"
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        
        return ax


    def plot_cumulative_ic(self, ax=None):
        """Plot cumulative IC."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate cumulative ICs
        cum_ic = np.cumsum(self.rankic_list)
        
        # Plot cumulative rank IC using rebalance dates
        ax.plot(self.rebalance_date[1:len(cum_ic)+1], cum_ic, label='Rank IC', color='blue')
        
        ax.set_title('Cumulative Rank IC')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative RankIC')
        ax.grid(True)
        ax.legend()
        
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        return ax

    def plot_cumulative_ic_interactive(self, ax=None):
        """Plot interactive cumulative IC."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure
        
        # Calculate cumulative ICs
        cum_ic = np.cumsum(self.rankic_list)
        
        # Plot using rebalance dates
        line = ax.plot(self.rebalance_date[1:len(cum_ic)+1], cum_ic, label='Rank IC', color='blue')[0]
        
        ax.set_title('Cumulative Rank IC')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative RankIC')
        ax.grid(True)
        ax.legend()
        
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # Add hover annotations
        cursor = mplcursors.cursor([line], hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.target
            date_idx = np.abs(matplotlib.dates.date2num(self.rebalance_date[1:len(cum_ic)+1]) - x).argmin()
            date_str = self.rebalance_date[date_idx + 1].strftime('%Y-%m-%d')
            sel.annotation.set_text(
                f"Date: {date_str}\n"
                f"Cum. RankIC: {y:.2f}\n"
                f"RankIC: {self.rankic_list[date_idx]:.3f}"
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        
        return ax
    
    def save_performance_summary_pdfs(self, output_dir='performance_plots', dpi=2400):
        """
        Save each row of the performance summary plot as a separate high-resolution PDF.
        
        Parameters:
        -----------
        output_dir : str
            Directory where PDF files will be saved
        dpi : int
            Resolution of the output PDFs
        """
        import os
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create individual figures for each row
        row_names = [
            'cumulative_returns',
            'excess_returns',
            'universe_excess_returns',
            'long_short_strategy',
            'long_universe_strategy',
            'short_universe_strategy',
            'long_benchmark_strategy',
            'long_short_stats',
            'long_universe_stats',
            'short_universe_stats',
            'long_benchmark_stats',
            'ic_stats',
            'rankic_series',
            'cumulative_ic',
            'turnover_metrics'
        ]
        
        for i, row_name in enumerate(row_names):
            fig = plt.figure(figsize=(16, 8))
            
            if i < 7:  # First 7 rows are plots
                if i == 0:
                    self.plot_cumulative_returns(plt.gca(), show_benchmark=True)
                elif i == 1:
                    self.plot_excess_returns(plt.gca())
                elif i == 2:
                    self.plot_universe_excess_returns(plt.gca())
                elif i >= 3 and i <= 6:
                    plot_types = ['long_short', 'long_universe', 'short_universe', 'long_benchmark']
                    self.plot_strategy_analysis(plot_types[i-3], plt.gca())
            
            elif i >= 7 and i <= 10:  # Monthly statistics tables
                strategies = ['long_short', 'long_universe', 'short_universe', 'long_benchmark']
                ax = plt.gca()
                ax.axis('off')
                
                stats_df = self.calculate_monthly_stats(strategies[i-7])
                stats_df = stats_df.round(2)
                
                table = ax.table(
                    cellText=stats_df.values,
                    colLabels=stats_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1]
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                
                # Set column widths
                month_cols = ['Jan%', 'Feb%', 'Mar%', 'Apr%', 'May%', 'Jun%', 
                            'Jul%', 'Aug%', 'Sep%', 'Oct%', 'Nov%', 'Dec%']
                for (row, col), cell in table._cells.items():
                    if col == 0:
                        cell.set_width(0.05)
                    elif stats_df.columns[col] in month_cols:
                        cell.set_width(0.04)
                    else:
                        cell.set_width(0.08)
                table.scale(2.5, 1.2)
                
                # Color header and alternating rows
                for r in range(len(stats_df.index) + 1):
                    for c in range(len(stats_df.columns)):
                        cell = table._cells.get((r, c))
                        if cell is not None:
                            if r == 0:
                                cell.set_facecolor('#ADD8E6')
                            elif r % 2:
                                cell.set_facecolor('#F0F0F0')
                
                plt.title(f"{strategies[i-7].replace('_', '-').title()} Monthly Statistics",
                         pad=20, fontsize=12, fontweight='bold')
            
            elif i == 11:  # IC stats table
                ax = plt.gca()
                ax.axis('off')
                
                ic_stats = self.calculate_ic_stats()
                if ic_stats is not None:
                    table = ax.table(
                        cellText=ic_stats.values,
                        colLabels=ic_stats.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1]
                    )
                    
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(2.5, 1.2)
                    
                    for (row, col), cell in table._cells.items():
                        if row == 0:
                            cell.set_facecolor('#ADD8E6')
                        cell.set_height(0.2)
                    
                    plt.title('IC Statistics', pad=20)
            
            elif i == 12:  # Rank IC series
                self.plot_rankic_series(plt.gca())
            
            elif i == 13:  # Cumulative IC
                self.plot_cumulative_ic(plt.gca())
            
            elif i == 14:  # Turnover and metrics
                gs = fig.add_gridspec(1, 2)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                
                self.plot_turnover(ax1)
                
                metrics = pd.DataFrame({
                    'Annual Return (%)': self.annual_returns,
                    'Sharpe Ratio': self.sharpe_ratios,
                    'Max Drawdown (%)': self.max_drawdowns
                }, index=[f'Group {i+1}' for i in range(self.group_num)])
                metrics.plot(kind='bar', ax=ax2)
                ax2.set_title('Performance Metrics by Group')
                ax2.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            pdf_path = os.path.join(output_dir, f'{row_name}.pdf')
            fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
        print(f"PDF files have been saved to {output_dir}/")