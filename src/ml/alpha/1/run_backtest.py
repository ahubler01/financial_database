#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from strategy import DataLoader, TradingStrategy
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run AAPL minute-bar trading strategy backtest')
    parser.add_argument('--train', type=str, required=True,
                      help='Training period start and end dates (YYYY-MM-DD,YYYY-MM-DD)')
    parser.add_argument('--val', type=str, required=True,
                      help='Validation period start and end dates (YYYY-MM-DD,YYYY-MM-DD)')
    parser.add_argument('--test', type=str, required=True,
                      help='Test period start and end dates (YYYY-MM-DD,YYYY-MM-DD)')
    parser.add_argument('--data', type=str, default='T2_engineered_features.csv',
                      help='Path to engineered features CSV file')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--initial-capital', type=float, default=1000000.0,
                      help='Initial capital for backtest (default: 1,000,000)')
    parser.add_argument('--target-freq', type=str, default='1H',
                      help='Target frequency for returns (1min, 5min, 1H, 1D)')
    return parser.parse_args()

def parse_date_range(date_range: str) -> tuple:
    start_date, end_date = date_range.split(',')
    return start_date.strip(), end_date.strip()

def generate_performance_report(metrics: Dict[str, Any], initial_capital: float, train_start: str, train_end: str, val_start: str, val_end: str, test_start: str, test_end: str, model_metrics: Dict[str, float], aapl_vol: float) -> str:
    report = [
        "Performance Report",
        "=================",
        "",
        "Trading Periods:",
        f"Training: {train_start} to {train_end}",
        f"Validation: {val_start} to {val_end}",
        f"Testing: {test_start} to {test_end}",
        "",
        "Market Context:",
        f"AAPL Annualized Volatility (Test Period): {aapl_vol:.2%}",
        "The test period was characterized by bearish market conditions, which may have",
        "impacted the strategy's performance. Consider adjusting the following parameters:",
        "1. Probability thresholds for long/short positions",
        "2. Position sizing methodology",
        "3. Risk management rules",
        "",
        "Model Performance (Validation Set):",
        f"Mean Squared Error (MSE): {model_metrics['mse']:.6f}",
        f"Root Mean Squared Error (RMSE): {model_metrics['rmse']:.6f}",
        "",
        "Key Performance Indicators:",
        f"1. Number of Trades: {len(metrics['trades'])}",
        f"2. Annualized Return: {metrics['annualized_return']:.2%}",
        f"3. Annualized Volatility: {metrics['volatility']:.2%}",
        f"4. Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
        f"5. Maximum Drawdown: {metrics['max_drawdown']:.2%}",
        f"6. Hit Rate: {metrics['hit_rate']:.2%}",
        f"7. Average PnL per Trade: ${metrics['avg_pnl_per_trade'] * initial_capital:.2f}",
        f"8. Annual Turnover: {metrics['turnover']:.2f}",
        "",
        "Strategy Overview:",
        "The strategy uses a LightGBM classifier to predict the direction of next-minute returns.",
        "Trades are executed when the model's probability exceeds predefined thresholds:",
        "- Long positions when p̂(y=+1) > 0.55",
        "- Short positions when p̂(y=+1) < 0.45",
        "- Flat otherwise",
        "",
        "Position sizing is volatility-scaled to target 1% annual volatility.",
        "The strategy includes 0.5bp slippage per round-trip trade.",
        "",
        f"Initial Capital: ${initial_capital:,.2f}",
        f"Final Portfolio Value: ${initial_capital * (1 + metrics['cumulative_return']):,.2f}",
        f"Total Return: {metrics['cumulative_return']:.2%}",
        "",
        "Risk Management:",
        "- Position sizing dynamically adjusts based on realized volatility",
        "- Maximum position size is capped by the target annual volatility",
        "- Trades are exited at t+1 or when signal flips, whichever comes first"
    ]
    return "\n".join(report)

def plot_performance(trades_df: pd.DataFrame, output_dir: Path):
    # Set style with dark background
    plt.style.use('dark_background')
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), facecolor='black')
    
    # Define the orange color
    ORANGE_COLOR = '#fb861b'
    
    # Plot cumulative returns
    trades_df['cumulative_return'] = (1 + trades_df['pnl']).cumprod()
    ax1.plot(trades_df['entry_time'], trades_df['cumulative_return'], color=ORANGE_COLOR, linewidth=1.5)
    ax1.set_title('Cumulative Returns', color='white')
    ax1.set_xlabel('Date', color='white')
    ax1.set_ylabel('Cumulative Return', color='white')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('black')
    
    # Plot drawdown
    rolling_max = trades_df['cumulative_return'].expanding().max()
    drawdown = (trades_df['cumulative_return'] - rolling_max) / rolling_max
    ax2.plot(trades_df['entry_time'], drawdown, color=ORANGE_COLOR, linewidth=1.5)
    ax2.set_title('Drawdown', color='white')
    ax2.set_xlabel('Date', color='white')
    ax2.set_ylabel('Drawdown', color='white')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('black')
    
    # Plot trade PnL distribution
    sns.histplot(trades_df['pnl'], bins=50, ax=ax3, color=ORANGE_COLOR)
    ax3.set_title('Trade PnL Distribution', color='white')
    ax3.set_xlabel('PnL', color='white')
    ax3.set_ylabel('Frequency', color='white')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('black')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_plots.png', facecolor='black', edgecolor='none')
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Parse date ranges
    train_start, train_end = parse_date_range(args.train)
    val_start, val_end = parse_date_range(args.val)
    test_start, test_end = parse_date_range(args.test)
    
    # Set target frequency in CONFIG
    from strategy import CONFIG
    CONFIG['TARGET_FREQUENCY'] = args.target_freq
    
    # Initialize data loader and load data
    data_loader = DataLoader(args.data)
    
    # Get data slices
    train_data = data_loader.get_data_slice(train_start, train_end)
    val_data = data_loader.get_data_slice(val_start, val_end)
    test_data = data_loader.get_data_slice(test_start, test_end)
    
    # Calculate AAPL volatility during test period
    aapl_returns = test_data['log_return']
    aapl_vol = aapl_returns.std() * np.sqrt(252 * 6.5 * 60)  # Annualize minute returns
    
    # Prepare features
    X_train, y_train = data_loader.prepare_features(train_data)
    X_val, y_val = data_loader.prepare_features(val_data)
    X_test, y_test = data_loader.prepare_features(test_data)
    
    # Initialize and train strategy
    strategy = TradingStrategy()
    strategy.train(X_train, y_train, X_val, y_val)
    
    # # # Perform hyperparameter tuning on validation set
    # logger.info("\nStarting hyperparameter tuning...")
    # best_params = strategy.tune_hyperparameters(X_val, y_val, val_data)
    # logger.info(f"\nBest parameters found: {best_params}")
    
    # Calculate model metrics on validation set
    val_predictions = strategy.model.predict(X_val)
    mse = np.mean((val_predictions - y_val) ** 2)
    rmse = np.sqrt(mse)
    model_metrics = {'mse': mse, 'rmse': rmse}
    
    # Save trained model
    model_path = output_dir / 'trained_model.pkl'
    strategy.save_model(str(model_path))
    
    # Generate signals and run backtest on test set
    test_signal = strategy.generate_signal(X_test)
    test_results = strategy.backtest(test_data, test_signal)
    
    # Create trades DataFrame and calculate cumulative return
    trades_df = pd.DataFrame(test_results['trades'])
    if not trades_df.empty:
        test_results['cumulative_return'] = (1 + trades_df['pnl']).prod() - 1
    else:
        test_results['cumulative_return'] = 0.0
    
    # Save trade-by-trade PnL if there are trades
    if not trades_df.empty:
        trades_df.to_csv(output_dir / 'trade_pnl.csv', index=False)
    
    # Generate and save performance report
    report = generate_performance_report(
        test_results, 
        args.initial_capital,
        train_start, train_end,
        val_start, val_end,
        test_start, test_end,
        model_metrics,
        aapl_vol
    )
    with open(output_dir / 'performance_report.txt', 'w') as f:
        f.write(report)
    
    # Save performance metrics
    metrics = {
        'annualized_return': test_results['annualized_return'],
        'volatility': test_results['volatility'],
        'sharpe_ratio': test_results['sharpe_ratio'],
        'max_drawdown': test_results['max_drawdown'],
        'hit_rate': test_results['hit_rate'],
        'avg_pnl_per_trade': test_results['avg_pnl_per_trade'],
        'turnover': test_results['turnover'],
        'cumulative_return': test_results['cumulative_return'],
        'initial_capital': args.initial_capital,
        'final_portfolio_value': args.initial_capital * (1 + test_results['cumulative_return'])
    }
    
    with open(output_dir / 'performance_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save performance plots if there are trades
    if not trades_df.empty:
        plot_performance(trades_df, output_dir)
    
    # Print results
    logger.info("\nBacktest Results:")
    logger.info("----------------")
    logger.info(report)

if __name__ == "__main__":
    main() 