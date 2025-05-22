# Financial Database Project

A high-frequency trading strategy implementation using machine learning to predict price movements in minute-by-minute data.

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Strategy Details](#strategy-details)
- [Configuration](#configuration)
- [Output](#output)
- [Documentation](#documentation)

## Overview

This project implements a machine learning-based trading strategy for AAPL using minute-bar data. The strategy:

- Uses LightGBM for probabilistic predictions
- Incorporates technical, market, macro, and sentiment features
- Employs dynamic position sizing based on volatility and prediction confidence
- Includes comprehensive risk management rules

## Data Sources

### Period Coverage: 2018-01-01 to 2020-12-31

1. **AAPL 1-Minute Data**

   - Source: [Kaggle AAPL Dataset](https://www.kaggle.com/datasets/deltatrup/aapl-1-minute-historical-stock-data-2006-2024)
   - Granularity: 1-minute bars
   - Fields: Open, High, Low, Close, Volume, Adj Close
   - Timezone: ET (Eastern Time)

2. **VIX Index**

   - Source: [CBOE VIX Historical Data](https://www.cboe.com/tradable_products/vix/vix_historical_data/)
   - Granularity: Daily
   - Fields: Open, High, Low, Close

3. **Macroeconomic Indicators** (Federal Reserve)
   | Indicator | Series ID | Frequency | Update Time (CDT) |
   |-----------|-----------|-----------|-------------------|
   | Business Confidence | BSCICP03USM665S | Monthly | 10th, 11:27 AM |
   | Net Exports | NETEXP | Quarterly | 27th, 8:03 AM |
   | Insured Unemployment | CCSA | Weekly | 24th, 7:48 AM |
   | Sticky CPI | CORESTICKM159SFRBATL | Monthly | 10th, 12:01 PM |
   | M1 Money Supply | WM1NS | Weekly | 22nd, 12:00 PM |
   | M2 Money Supply | WM2NS | Weekly | 22nd, 12:00 PM |
   | Fed Funds Rate | FEDFUNDS | Monthly | 1st, 3:17 PM |

4. **Sentiment Data**
   - Source: [Reddit Stock Posts Dataset](https://www.kaggle.com/datasets/injek0626/reddit-stock-related-posts)
   - Coverage: AAPL, GME, MCD, MSFT, NFLX, NVDA, TSLA
   - Files: posts.csv, stock_index.csv, subreddit_subscribers.csv

## Requirements

- Python 3.8+
- pandas
- numpy
- lightgbm
- scikit-learn
- matplotlib
- seaborn
- optuna

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/financial_database.git
   cd financial_database
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Processing Pipeline**:

   ```bash
   # Run T1 (Data Integration)
   python src/data_processing/T1.py

   # Run T2 (Feature Engineering)
   python src/data_processing/T2_feature_engineering.py \
       --train-period "2018-01-01,2019-06-01" \
       --val-period "2019-06-01,2019-12-31" \
       --test-period "2020-01-01,2020-12-31" \
       --freq "1min" \
       --output-dir "data"
   ```

2. **Run Trading Strategy**:
   ```bash
   python src/ml/alpha/run_backtest.py \
       --train "2018-01-01,2019-06-01" \
       --val "2019-06-01,2019-12-31" \
       --test "2020-01-01,2020-12-31" \
       --data "T2_engineered_features.csv" \
       --output-dir "results" \
       --initial-capital 1000000 \
       --target-freq "1H"
   ```

## Strategy Details

### Model

- **Algorithm**: LightGBM classifier
- **Target Variable**: Sign of next-minute returns (with 5bp threshold)
- **Features**: Technical, market, macro, and sentiment indicators

### Trading Rules

- **Entry**:

  - Long: p̂(y=+1) > 0.62
  - Short: p̂(y=+1) < 0.38
  - Neutral: 0.38 ≤ p̂(y=+1) ≤ 0.62

- **Position Sizing**:

  - Base size: TARGET_ANNUAL_VOL / realized_vol
  - Confidence multiplier: |prob - 0.5| \* 2
  - Maximum position: 20.0 units

- **Exit Conditions**:
  - Time-based: t+10 minutes
  - Signal-based: When prediction crosses neutral zone
  - Whichever comes first

## Configuration

Key parameters can be modified in `src/ml/alpha/1/strategy.py`:

```python
CONFIG = {
    'EPSILON': 0.0005,           # 5bp threshold
    'LONG_THRESHOLD': 0.62,      # Long entry threshold
    'SHORT_THRESHOLD': 0.38,     # Short entry threshold
    'TARGET_ANNUAL_VOL': 0.45,   # Target volatility
    'SLIPPAGE': 0.0001,         # 1bp per round-trip
    'EXIT_HORIZON': 60*10,      # 10-minute max hold
    'MAX_POSITION': 20.0,       # Position cap
    'POSITION_MULTIPLIER': 15.5  # Size multiplier
}
```

## Output

Results are saved in the specified output directory:

- `trained_model.pkl`: Serialized model
- `trade_pnl.csv`: Trade-by-trade results
- `performance_metrics.json`: Strategy metrics
- `performance_plots.png`: Visual analysis
- `performance_report.txt`: Detailed report

## Documentation

- [Data Pipeline Documentation](DATA_PIPELINE.md)
- [Strategy Documentation](src/ml/alpha/1/strategy.py)
- [Feature Engineering Details](src/data_processing/T2_feature_engineering.py)

## Performance Results

Latest backtest results for period 2020-01-01 to 2020-12-31:

### Trading Statistics

| Metric                | Value   |
| --------------------- | ------- |
| Number of Trades      | 1,134   |
| Hit Rate              | 54.32%  |
| Average PnL per Trade | $108.86 |
| Annual Turnover       | -       |

### Returns and Risk

| Metric                | Value         |
| --------------------- | ------------- |
| Initial Capital       | $1,000,000.00 |
| Final Portfolio Value | $1,131,069.81 |
| Total Return          | 13.11%        |
| Annualized Return     | 1069.86%      |
| Annualized Volatility | 16.99%        |
| Sharpe Ratio          | 10.46         |
| Maximum Drawdown      | -0.89%        |

### Model Performance (Validation Set)

| Metric                         | Value  |
| ------------------------------ | ------ |
| Root Mean Squared Error (RMSE) | 4.6856 |

Note: The test period (2020) was characterized by high market volatility and bearish conditions, which influenced strategy performance. Risk management parameters may need adjustment for different market regimes.
