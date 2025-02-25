# Gnosiom - an open source end-to-end solution to factor investment research in US stocks

## Project Vision

We aims to be an open-source framework for conducting rigorous daily-level stock backtesting. This framework bridges that gap by enabling accurate, daily-level backtesting using high-quality WRDS data.

## Workflow

To run this engine, here is the workflow

Step 1: Data Import
We start by running a notebook (e.g., create_dsf_v2.ipynb) that pulls raw financial data from an online source (using the WRDS Python API) and saves it into our local database (DuckDB). 

Step 2: Define Your Factor
Next, you open a Python file (FactorCalculator.py) where you write a custom SQL query. This query calculates a “factor” (e.g. ROE，PB ratio) that will be used later in the analysis. 

Step 3: Calculate Daily Factor Data
Then, you run another notebook (create_factor_parquet.ipynb) that applies your custom factor calculation to generate daily data. For example, if your factor is ROE, you’ll end up with a daily snapshot of ROE values across different companies. 

Step 4: Factor Analysis
After that, you analyze the computed factor by grouping the data (using factor_analysis.ipynb). You can find  more results from the slides

Step 5: Backtest the Trading Strategy
Finally, you run a notebook (backtest_portfolio.ipynb) that simulates a trading strategy based on your factor. This backtesting step includes realistic details like taxes, commissions, slippage, and other market factors. You can also find more results from the slides




### Why Daily Backtesting?

While monthly backtesting dominates academic literature and many open-source implementations, we believe this approach doesn't adequately reflect real-world trading practices. Here's why daily backtesting matters:

1. **Realistic Trading Frequency**: In actual investment practice, portfolio rebalancing often occurs at daily frequencies rather than monthly. Our own trading experience confirms that waiting for month-end to make investment decisions can miss significant opportunities and risks.

2. **Precise Performance Measurement**: Daily-level analysis provides a more granular and accurate view of strategy performance, capturing intra-month dynamics that monthly data would smooth over.

3. **Better Risk Management**: Daily monitoring and rebalancing capabilities allow for more responsive risk management, better reflecting how real portfolios are managed.

### Why WRDS (CRSP and Compustat)?

Our choice of WRDS data, specifically CRSP and Compustat, was driven by two critical requirements for accurate backtesting:

1. **Point-in-Time Financial Data**: 
  - Financial statements can be restated over time
  - The financial data visible to an investor on any historical date may differ from what we see today
  - Compustat's point-in-time data ensures our backtests only use information that was actually available on each historical date
  - Many other databases provide only single snapshots of financial data per period, either ignoring restatements or using only final numbers, leading to backtest inaccuracies

2. **Accurate Price Return Calculations**:
  - CRSP provides comprehensive handling of:
    - Stock splits
    - Dividend adjustments
    - Delisting returns
  - These factors significantly impact actual investment returns and cash positions
  - CRSP's careful treatment of these events helps our backtests better approximate real-world trading outcomes

### Data Accessibility

While WRDS access requires a subscription, many academic institutions provide access to their students and faculty. If you're a university student or researcher, you likely already have access to these databases through your institution.

### Technical Implementation

Our framework utilizes:
1. **WRDS Python API**: For efficient data retrieval from CRSP and Compustat
2. **DuckDB**: A high-performance analytical database for local data storage and processing
3. **SQL-Based Factor Construction**: Clear and efficient factor calculations using SQL queries
4. **Parquet File Storage**: Efficient storage and retrieval of calculated factors

This technical stack enables:
- Efficient processing of large datasets
- Clear and maintainable factor definitions
- Fast daily-level backtesting
- Reproducible research results

