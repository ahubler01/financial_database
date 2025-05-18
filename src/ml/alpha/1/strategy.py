import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Dict, List, Any
import pickle
from datetime import datetime
import os
import logging
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
CONFIG = {
    'EPSILON': 0.0005,  # 5bp threshold for label definition (increased from 1bp)
    'LONG_THRESHOLD': 0.62,  # Probability threshold for long positions
    'SHORT_THRESHOLD': 0.38,  # Probability threshold for short positions
    'TARGET_ANNUAL_VOL': 0.45,  # 1% target annual volatility
    'SLIPPAGE': 0.0001,   # assume 1bp per round‐trip
    'EXIT_HORIZON': 60*10,  # minutes
    'MAX_POSITION': 20.0,  # maximum position size
    'POSITION_MULTIPLIER': 15.5,  # multiplier to increase position size
    'TARGET_FREQUENCY': '1h',  # frequency for target variable ('1min', '5min', '1H', '1D')
    'MODEL_PARAMS': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data_in_leaf': 20,
        'min_gain_to_split': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'max_depth': 5,
        'scale_pos_weight': 1.0
    }
}

class DataLoader:
    def __init__(self, data_path: str):
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.data_dir = os.path.join(self.project_root, 'data')
        
        # Load pre-split dataframes
        self.train_data = pd.read_csv(os.path.join(self.data_dir, 'train_features.csv'))
        self.val_data = pd.read_csv(os.path.join(self.data_dir, 'val_features.csv'))
        self.test_data = pd.read_csv(os.path.join(self.data_dir, 'test_features.csv'))
        
        # Convert timestamps and set index
        for df in [self.train_data, self.val_data, self.test_data]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # logger.info data shapes and date ranges
        logger.info("\nData Shapes:")
        logger.info(f"Train data: {self.train_data.shape}")
        logger.info(f"Validation data: {self.val_data.shape}")
        logger.info(f"Test data: {self.test_data.shape}")
        
        logger.info("\nDate Ranges:")
        logger.info(f"Train: {self.train_data.index.min()} to {self.train_data.index.max()}")
        logger.info(f"Validation: {self.val_data.index.min()} to {self.val_data.index.max()}")
        logger.info(f"Test: {self.test_data.index.min()} to {self.test_data.index.max()}")
        
    def validate_dates(self, start_date: str, end_date: str, period: str) -> None:
        """Validate that the requested date range matches the data"""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        if period == 'train':
            data_start = self.train_data.index.min()
            data_end = self.train_data.index.max()
        elif period == 'val':
            data_start = self.val_data.index.min()
            data_end = self.val_data.index.max()
        elif period == 'test':
            data_start = self.test_data.index.min()
            data_end = self.test_data.index.max()
        else:
            raise ValueError(f"Invalid period: {period}")
        
        if start != data_start or end != data_end:
            raise ValueError(
                f"Date range mismatch for {period} period:\n"
                f"Requested: {start} to {end}\n"
                f"Available: {data_start} to {data_end}"
            )
    
    def get_data_slice(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get the appropriate data slice based on the date range"""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Determine which dataset to use based on the date range
        if start >= self.train_data.index.min() and end <= self.train_data.index.max():
            self.validate_dates(start_date, end_date, 'train')
            return self.train_data.copy()
        elif start >= self.val_data.index.min() and end <= self.val_data.index.max():
            self.validate_dates(start_date, end_date, 'val')
            return self.val_data.copy()
        elif start >= self.test_data.index.min() and end <= self.test_data.index.max():
            self.validate_dates(start_date, end_date, 'test')
            return self.test_data.copy()
        else:
            raise ValueError(
                f"Date range {start_date} to {end_date} does not match any dataset:\n"
                f"Train: {self.train_data.index.min()} to {self.train_data.index.max()}\n"
                f"Validation: {self.val_data.index.min()} to {self.val_data.index.max()}\n"
                f"Test: {self.test_data.index.min()} to {self.test_data.index.max()}"
            )
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable based on configured frequency"""
        # Create aggregated returns based on target frequency
        if CONFIG['TARGET_FREQUENCY'] != '1min':
            # Resample close prices to target frequency
            resampled_close = df['Close'].resample(CONFIG['TARGET_FREQUENCY']).last()
            # Calculate log returns at target frequency
            price_ratio = resampled_close / resampled_close.shift(1)
            price_ratio = price_ratio.replace([0, np.inf, -np.inf], np.nan)
            price_ratio = price_ratio.ffill().fillna(1.0)
            price_ratio = price_ratio + 1e-10
            agg_log_returns = np.log(price_ratio)
            
            # Reindex back to original frequency and forward fill
            df['agg_log_return'] = agg_log_returns.reindex(df.index, method='ffill')
            
            # Apply 3-period smoothing to returns
            df['smoothed_return'] = df['agg_log_return'].rolling(window=3).mean()
            
            # Create binary label: 1 if next period return is positive and above epsilon, 0 otherwise
            df['label'] = (df['smoothed_return'].shift(-1) > CONFIG['EPSILON']).astype(int)
        else:
            # Original 1-minute frequency logic
            df['label'] = (df['log_return'].shift(-1) > CONFIG['EPSILON']).astype(int)
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['label', 'log_return', 'agg_log_return', 'smoothed_return']]
        X = df[feature_cols]
        y = df['label']
        
        return X, y

class TradingStrategy:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.best_params = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model with default parameters
        self.model = lgb.train(
            CONFIG['MODEL_PARAMS'],
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
        )
    
    def generate_signal(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        prob_pos = self.model.predict(X)
        signal = pd.Series(0, index=X.index)
        
        signal[prob_pos > CONFIG['LONG_THRESHOLD']] = 1
        signal[prob_pos < CONFIG['SHORT_THRESHOLD']] = -1
        
        return signal
    
    def calculate_position_size(self, df: pd.DataFrame) -> pd.Series:
        # Scale 3-block realized vol to annual
        annualized_vol = df['realized_vol_3block'] * np.sqrt(252 * 6.5 * 60)  # 6.5 hours * 60 minutes
        
        # Calculate base position size and apply multiplier
        base_position_size = CONFIG['TARGET_ANNUAL_VOL']/annualized_vol
        position_size = base_position_size * CONFIG['POSITION_MULTIPLIER']
        
        # Apply maximum position limit
        position_size = np.minimum(position_size, CONFIG['MAX_POSITION'])
        
        # Add conviction scaling - increase position size when vol is lower
        vol_percentile = df['realized_vol_3block'].rolling(window=30).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        conviction_multiplier = 1.0 + (1.0 - vol_percentile) * 0.5  # Up to 50% more when vol is low
        
        # Apply conviction multiplier
        position_size = position_size * conviction_multiplier
        
        # Ensure max position constraint is still respected
        position_size = np.minimum(position_size, CONFIG['MAX_POSITION'])
        
        return position_size
    
    def backtest(self, df: pd.DataFrame, signal: pd.Series) -> Dict:
        # Initialize position tracking
        exit_horizon = CONFIG.get('EXIT_HORIZON', 1)
        position = pd.Series(0, index=df.index)
        trades = []
        current_position = 0
        entry_idx = None
        
        # Calculate position sizes
        position_size = self.calculate_position_size(df)
        
        # Run backtest
        for i in range(len(df)):
            # 1) Horizon-based exit
            if current_position != 0 and entry_idx is not None and (i - entry_idx) >= exit_horizon:
                exit_price = df['Close'].iloc[i]
                entry_price = trades[-1]['adjusted_entry_price'] if 'adjusted_entry_price' in trades[-1] else trades[-1]['entry_price']
                # Apply slippage to the exit price based on direction
                slippage_adjustment = CONFIG['SLIPPAGE'] * exit_price * (-current_position) # Negative for buys, positive for sells
                adjusted_exit_price = exit_price + slippage_adjustment
                pnl = current_position * trades[-1]['size'] * (adjusted_exit_price - entry_price) / entry_price
                trades[-1].update({
                    'exit_time': df.index[i],
                    'exit_price': exit_price,
                    'adjusted_exit_price': adjusted_exit_price,
                    'pnl': pnl
                })
                current_position = 0
                entry_idx = None
            
            # 2) Signal-based change (if different from current, and not just our forced exit)
            if signal.iloc[i] != current_position:
                # Close existing position if any (and not already closed by horizon)
                if current_position != 0 and entry_idx is not None:
                    exit_price = df['Close'].iloc[i]
                    entry_price = trades[-1]['adjusted_entry_price'] if 'adjusted_entry_price' in trades[-1] else trades[-1]['entry_price']
                    # Apply slippage to the exit price based on direction
                    slippage_adjustment = CONFIG['SLIPPAGE'] * exit_price * (-current_position) # Negative for buys, positive for sells
                    adjusted_exit_price = exit_price + slippage_adjustment
                    pnl = current_position * trades[-1]['size'] * (adjusted_exit_price - entry_price) / entry_price
                    trades[-1].update({
                        'exit_time': df.index[i],
                        'exit_price': exit_price,
                        'adjusted_exit_price': adjusted_exit_price,
                        'pnl': pnl
                    })
                
                # Open new position if signal is not flat
                if signal.iloc[i] != 0:
                    entry_price = df['Close'].iloc[i]
                    # Apply slippage to the entry price based on direction
                    slippage_adjustment = CONFIG['SLIPPAGE'] * entry_price * signal.iloc[i]  # Positive for buys, negative for sells
                    adjusted_entry_price = entry_price + slippage_adjustment
                    trades.append({
                        'entry_time': df.index[i],
                        'entry_price': entry_price, # Store original price for reference
                        'adjusted_entry_price': adjusted_entry_price, # Store slippage-adjusted price for PnL calc
                        'direction': signal.iloc[i],
                        'size': position_size.iloc[i],
                        'pnl': 0.0  # will be set on exit
                    })
                    entry_idx = i
                
                current_position = signal.iloc[i]
            
            position.iloc[i] = current_position
        
        # (rest of your metrics calculation stays the same...)
        if not trades:
            return {
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'hit_rate': 0,
                'avg_pnl_per_trade': 0,
                'turnover': 0,
                'trades': []
            }
        
        returns = pd.Series([t['pnl'] for t in trades], index=[t['entry_time'] for t in trades])
        annualized_return = returns.mean() * 252 * 6.5 * 60
        volatility        = returns.std()  * np.sqrt(252 * 6.5 * 60)
        sharpe_ratio      = annualized_return - 4.02 / (volatility*100) if volatility != 0 else 0
        cum_returns       = (1 + returns).cumprod()
        rolling_max       = cum_returns.expanding().max()
        max_drawdown      = ((cum_returns - rolling_max) / rolling_max).min()
        hit_rate          = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
        avg_pnl_per_trade = returns.mean()
        turnover          = sum(abs(t['size']) for t in trades) / len(trades)
        
        return {
            'annualized_return': annualized_return,
            'volatility':        volatility,
            'sharpe_ratio':      sharpe_ratio,
            'max_drawdown':      max_drawdown,
            'hit_rate':          hit_rate,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'turnover':          turnover,
            'trades':            trades
        }

    
    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained yet")
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load_model(cls, path: str) -> 'TradingStrategy':
        strategy = cls()
        with open(path, 'rb') as f:
            strategy.model = pickle.load(f)
        return strategy 

    def objective(self, trial, X_val: pd.DataFrame, y_val: pd.Series, df_val: pd.DataFrame) -> float:
        """Objective function for Optuna optimization"""
        # Define parameter ranges
        long_threshold = trial.suggest_float('LONG_THRESHOLD', 0.6, 0.9)
        short_threshold = trial.suggest_float('SHORT_THRESHOLD', 0.1, 0.4)
        target_vol = trial.suggest_float('TARGET_ANNUAL_VOL', 0.01, 0.2)
        exit_horizon = trial.suggest_int('EXIT_HORIZON', 1, 15) * 60  # Convert to minutes
        epsilon = trial.suggest_float('EPSILON', 0.0001, 0.005)
        
        # Ensure valid threshold combination
        if short_threshold >= long_threshold:
            return float('-inf')
        
        # Update CONFIG
        CONFIG.update({
            'LONG_THRESHOLD': long_threshold,
            'SHORT_THRESHOLD': short_threshold,
            'TARGET_ANNUAL_VOL': target_vol,
            'EXIT_HORIZON': exit_horizon,
            'EPSILON': epsilon
        })
        
        # Generate signals and backtest
        signal = self.generate_signal(X_val)
        results_dict = self.backtest(df_val, signal)
        
        trades_df = pd.DataFrame(results_dict['trades'])
        
        # Calculate total return
        if not trades_df.empty:
            total_return = (1 + trades_df['pnl']).prod() - 1
        else:
            total_return = 0
        
        # Log trial results
        logger.info(f"\nTrial {trial.number}:")
        logger.info(f"Parameters:")
        logger.info(f"  Long Threshold: {long_threshold:.2f}")
        logger.info(f"  Short Threshold: {short_threshold:.2f}")
        logger.info(f"  Target Vol: {target_vol:.3%}")
        logger.info(f"  Exit Horizon: {exit_horizon/60:.0f} minutes")
        logger.info(f"  Epsilon: {epsilon:.4f}")
        logger.info(f"Results:")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Number of Trades: {len(results_dict['trades'])}")
        logger.info(f"  Hit Rate: {results_dict['hit_rate']:.2%}")
        logger.info(f"  Sharpe Ratio: {results_dict['sharpe_ratio']:.2f}")
        
        # Store trial results
        trial.set_user_attr('num_trades', len(results_dict['trades']))
        trial.set_user_attr('hit_rate', results_dict['hit_rate'])
        trial.set_user_attr('sharpe_ratio', results_dict['sharpe_ratio'])
        
        return total_return
    
    def tune_hyperparameters(self, X_val: pd.DataFrame, y_val: pd.Series, df_val: pd.DataFrame) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        logger.info("\nStarting hyperparameter optimization with Optuna")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_val, y_val, df_val),
            n_trials=100,
            timeout=3600  # 1 hour timeout
        )
        
        # Get best parameters
        best_params = study.best_params
        best_params['EXIT_HORIZON'] = best_params['EXIT_HORIZON'] * 60  # Convert back to minutes
        
        # Log best trial
        logger.info("\nBest trial:")
        logger.info(f"  Value: {study.best_value:.2%}")
        logger.info(f"  Params: {best_params}")
        logger.info(f"  Number of Trades: {study.best_trial.user_attrs['num_trades']}")
        logger.info(f"  Hit Rate: {study.best_trial.user_attrs['hit_rate']:.2%}")
        logger.info(f"  Sharpe Ratio: {study.best_trial.user_attrs['sharpe_ratio']:.2f}")
        
        # Update CONFIG with best parameters
        CONFIG.update(best_params)
        self.best_params = best_params
        
        return best_params 