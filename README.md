# financial_database

Big Data Group Project

# Data

### Period

01/01/2018 - 12/12/2020

### Stock price

One-minute granularity, ET time

Tickers:

- aapl
- amzn
- tsla
- nvda
- msft
- spx
- vix, https://www.cboe.com/tradable_products/vix/vix_historical_data/

Spy: Issue Accessing Oneminutedata.com

### Macroeconomics indicators

- composite_business_confidence_2018_2020\_\_monthly_fed, updated on 10th of the month at 11:27 AM CDT
- net_exports_2018_2020\_\_quarterly_fed, updated on 27th of the month at 8:03AM CDT
- insured_unemployment_2018_2020\_\_weekly_fed, updated on the 24th of the month at 7:48 AM CDT
- sticky_cpi_2018_2020\_\_monthly_fed, updated on the 10 of the month at 12:01 PM CDT
- m2_2018_2020\_\_weekly_fed, updated on the 22 of the month at 12:00 PM CDT
- m1_2018_2020\_\_weekly_fed, updated on the 22 of the month at 12:00 PM CDT
- effective_rates_2018_2020_monthly_fed, update on the 1st of the month at 3:17PM CDT

'CORESTICKM159SFRBATL': sticky_cpi_2018_2020
'NETEXP': net_exports_2018_2020
'FEDFUNDS': effective_rates_2018_2020_monthly_fed
'WM1NS': m1_2018_2020
'WM2NS': m2_2018_2020
'BSCICP03USM665S': composite_business_confidence_2018_2020
'CCSA': insured_unemployment_2018_2020

Would need a conversion from CDT to ET

### Sentiment data (Reddit)

https://www.kaggle.com/datasets/injek0626/reddit-stock-related-posts?select=posts.csv

- posts.csv (AAPL, GME, MCD, MSFT, NFLX, NVDA, and TSLA from 2018 to 2022)
- stock_index.csv (maps post with stock symbol and utc time)
- subreddit_subscribers.csv (31st July 2023)

# Data Transformation Pipeline

The data transformation pipeline (`src/data/T1.py`) performs the following operations:

1. **Data Loading and Merging**

   - Loads stock data (AAPL and VIX) and merges them
   - Loads all macro-economic indicators and merges them
   - Loads Reddit data and filters for the period 2018-2020
   - Aggregates Reddit posts to daily frequency using sum aggregation for post counts and mean aggregation for sentiment scores
   - Merges all data sources on timestamp using `merge_asof` with backward direction to preserve all time points
   - Handles timezone differences by converting all timestamps to ET (Eastern Time)
   - Applies forward-fill (ffill) after merging to handle any remaining gaps

2. **Data Cleaning**

   - Removes rows with all NaN values
   - Applies forward-fill (ffill) to stock and macro data to handle missing values
   - Removes any remaining NaN values after processing

3. **Data Normalization**

   - Applies winsorization to all numeric columns (5% on both tails) to handle outliers
   - Applies min-max normalization to all numeric columns to scale values between 0 and 1

4. **Output**
   - Saves the processed data to `data/processed_data_2018_2020.csv`
   - Logs information about the processing steps and final data shape

The pipeline ensures that all data is properly aligned in time and normalized for further analysis.

## Quant ML Trading Strategy

### Overview

This repository implements a machine learning-based trading strategy for AAPL using minute-bar data. The strategy uses engineered features to predict the direction of next-minute returns and generates trading signals based on the model's predictions.

**Dependent Variable**: Sign of next-minute returns (with 1bp threshold)
**Algorithm**: LightGBM classifier with hyperparameter tuning
**Feature Universe**: Engineered features from minute-bar data (see `T2_engineered_features.csv`)

### Label Definition

The strategy predicts the sign of next-minute returns:

- yₜ = sign(log_returnₜ₊₁) with threshold ε=0.0001 (1bp)
- Returns below the threshold are considered flat (0)

### Model

The strategy uses a LightGBM classifier with the following characteristics:

- Binary classification objective
- Hyperparameter tuning via grid search on validation set
- Early stopping to prevent overfitting
- Feature importance analysis available

### Trading Rules

**Entry Conditions**:

- Long position: p̂(y=+1) > 0.55
- Short position: p̂(y=+1) < 0.45
- Flat position: otherwise

**Exit Conditions**:

- Exit at t+1 or when signal flips, whichever comes first
- Slippage assumption: 0.5bp per round-trip

### Position Sizing

- Target annual volatility: 1%
- Position size: 1% / realized_vol_3blockₜ (scaled to p.a.)
- Volatility scaling based on 3-block realized volatility

### Backtest Results

| Metric            | Value |
| ----------------- | ----- |
| Annualized Return | TBD   |
| Volatility        | TBD   |
| Sharpe Ratio      | TBD   |
| Max Drawdown      | TBD   |
| Hit Rate          | TBD   |
| Avg PnL per Trade | TBD   |
| Turnover          | TBD   |

### How to Run

1. Ensure you have the required dependencies installed:

   ```bash
   pip install pandas numpy lightgbm scikit-learn
   ```

2. Run the backtest with specified date ranges:

   ```bash
   python alpha/run_backtest.py \
     --train "2020-01-01,2020-12-31" \
     --val "2021-01-01,2021-12-31" \
     --test "2022-01-01,2022-12-31" \
     --data T2_engineered_features.csv \
     --output-dir results
   ```

3. Results will be saved in the specified output directory:
   - `trained_model.pkl`: Serialized LightGBM model
   - `trade_pnl.csv`: Trade-by-trade PnL data
   - `performance_metrics.json`: Summary performance metrics

# High-Frequency Trading Strategy

This repository implements a high-frequency trading strategy using machine learning to predict price movements in minute-by-minute data.

## Strategy Overview

The strategy uses a LightGBM classifier to predict the direction of next-minute returns. Key components include:

1. **Feature Engineering**: Technical indicators and price-based features are calculated from minute-by-minute data
2. **Model Training**: A LightGBM classifier is trained to predict the probability of positive returns
3. **Signal Generation**: Trading signals are generated based on probability thresholds
4. **Position Sizing**: Positions are sized based on volatility targeting
5. **Risk Management**: Multiple exit conditions and position limits are implemented

## Configuration Parameters

### Model Parameters

```python
'MODEL_PARAMS': {
    'objective': 'binary',        # Binary classification task
    'metric': 'binary_logloss',   # Evaluation metric
    'boosting_type': 'gbdt',      # Gradient Boosting Decision Tree
    'num_leaves': 31,            # Number of leaves in each tree
    'learning_rate': 0.05,       # Learning rate for gradient boosting
    'feature_fraction': 0.9,     # Fraction of features to use in each iteration
    'bagging_fraction': 0.8,     # Fraction of data to use in each iteration
    'bagging_freq': 5,           # Frequency of bagging
    'verbose': -1                # Suppress LightGBM output
}
```

### Trading Parameters

```python
'EPSILON': 0.0001,              # 1bp threshold for label definition
'LONG_THRESHOLD': 0.25,         # Probability threshold for long positions
'SHORT_THRESHOLD': 0.75,        # Probability threshold for short positions
'TARGET_ANNUAL_VOL': 0.001,     # 1% target annual volatility
'SLIPPAGE': 0.0001,             # 1bp slippage per round-trip trade
'EXIT_HORIZON': 60*5,           # 5-minute maximum holding period
'MAX_POSITION': 15.0,           # Maximum position size
'SIGNAL_FREQUENCY': '1D',       # Frequency for signal aggregation
'PROB_AGG_METHOD': 'mean'       # Method for probability aggregation
```

## Signal Generation Process

1. **Minute-by-Minute Prediction**:

   - The model generates probability predictions for each minute
   - These probabilities represent the likelihood of a positive return in the next minute

2. **Probability Aggregation**:

   - Probabilities are aggregated to the specified frequency (default: daily)
   - Available aggregation methods:
     - `mean`: Average probability over the period
     - `median`: Middle probability over the period
     - `max`: Most bullish probability in the period
     - `min`: Most bearish probability in the period

3. **Signal Creation**:
   - Long positions when aggregated probability > 0.25
   - Short positions when aggregated probability < 0.75
   - Flat position otherwise

## Position Sizing and Risk Management

1. **Volatility Targeting**:

   - Position sizes are scaled to target 1% annual volatility
   - Formula: `position_size = min(TARGET_ANNUAL_VOL/annualized_vol, MAX_POSITION)`
   - Annualized volatility is calculated from 3-block realized volatility

2. **Exit Conditions**:

   - Time-based exit: Positions are automatically closed after 5 minutes
   - Signal-based exit: Positions are closed when the signal flips direction
   - Whichever condition occurs first triggers the exit

3. **Risk Controls**:
   - Maximum position size cap (15.0 units)
   - Slippage costs included in PnL calculations
   - Dynamic position sizing based on realized volatility

## Performance Metrics

The strategy tracks several key performance indicators:

1. Number of Trades
2. Annualized Return
3. Annualized Volatility
4. Sharpe Ratio
5. Maximum Drawdown
6. Hit Rate (percentage of profitable trades)
7. Average PnL per Trade
8. Annual Turnover

## Usage

1. **Data Preparation**:

   - Ensure data is in the correct format with required features
   - Split data into training, validation, and test periods

2. **Model Training**:

   - Train the model using the training and validation sets
   - Adjust model parameters as needed

3. **Backtesting**:

   - Run backtest with specified parameters
   - Analyze performance metrics and adjust strategy parameters

4. **Parameter Optimization**:
   - Experiment with different signal frequencies
   - Try different probability aggregation methods
   - Adjust position sizing and risk parameters

## Example Configuration

```python
# Daily signals using mean probability
CONFIG = {
    'SIGNAL_FREQUENCY': '1D',
    'PROB_AGG_METHOD': 'mean',
    'LONG_THRESHOLD': 0.25,
    'SHORT_THRESHOLD': 0.75,
    'TARGET_ANNUAL_VOL': 0.001,
    'EXIT_HORIZON': 60*5
}
```

## Notes

- The strategy is designed for high-frequency trading with quick entry and exit
- Asymmetric thresholds suggest different risk tolerances for long vs short positions
- Performance may vary significantly in different market conditions
- Consider adjusting parameters based on market regime and volatility environment

## Model Enhancements

### Threshold Optimization

The strategy now includes an automated threshold optimization process that:

- Performs a grid search over short (0.1-0.4) and long (0.6-0.9) thresholds
- Evaluates each threshold pair on the validation set using Sharpe ratio
- Automatically updates the configuration with optimal thresholds
- Results are stored in the strategy's `optimal_thresholds` attribute

### Confidence-Based Position Sizing

Position sizing now incorporates model confidence:

- Base position size is calculated as: `TARGET_ANNUAL_VOL / annualized_vol`
- Confidence score is computed as: `|prob_pos - 0.5| * 2` (maps [0.5-1] → [0-1])
- Final position size is: `min(base_size * confidence, MAX_POSITION)`
- This results in larger positions for high-confidence predictions and smaller positions for uncertain ones

### Target Smoothing

The target variable has been enhanced with:

- 3-period rolling mean smoothing of returns
- Increased epsilon threshold to 5 basis points (from 1bp)
- Benefits:
  - Reduces noise in the target variable
  - Captures more meaningful price movements
  - Improves model stability
  - Better aligns with actual trading conditions

### Usage

To run the enhanced strategy:

```bash
python src/ml/alpha/run_backtest.py \
    --train 2018-01-01,2019-06-01 \
    --val 2019-06-01,2019-12-31 \
    --test 2020-01-01,2020-12-31 \
    --data T2_engineered_features.csv \
    --output-dir results \
    --initial-capital 1000000 \
    --target-freq 1H
```

The strategy will:

1. Load and prepare the data with smoothed targets
2. Train the model with optimized parameters
3. Perform threshold optimization on the validation set
4. Run the backtest with confidence-based position sizing
5. Generate performance reports and plots
