import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, 
                 train_period: Tuple[str, str],
                 val_period: Tuple[str, str],
                 test_period: Tuple[str, str],
                 freq: str = '1min',
                 data_dir: str = "data"
                ):
        """Initialize the feature engineering class
        
        Args:
            train_period: Tuple of (start_date, end_date) for training period
            val_period: Tuple of (start_date, end_date) for validation period
            test_period: Tuple of (start_date, end_date) for test period
            freq: Frequency of data (default: '1min')
            data_dir: Directory containing input data files
        """
        self.data_dir = Path(data_dir)
        self.t1_merged = None
        self.sentiment_data = None
        self.stock_mapping = None
        self.features = None
        self.freq = freq
        
        # Convert string dates to Timestamps
        self.train_start = pd.Timestamp(train_period[0])
        self.train_end = pd.Timestamp(train_period[1])
        self.val_start = pd.Timestamp(val_period[0])
        self.val_end = pd.Timestamp(val_period[1])
        self.test_start = pd.Timestamp(test_period[0])
        self.test_end = pd.Timestamp(test_period[1])
        
        # Compute horizons
        self.horizons = {
            'train': self._period_steps(self.train_start, self.train_end),
            'val': self._period_steps(self.val_start, self.val_end),
            'test': self._period_steps(self.test_start, self.test_end)
        }
        
    def _period_steps(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Compute number of steps between start and end dates at given frequency
        
        Args:
            start: Start timestamp
            end: End timestamp
            
        Returns:
            Number of steps between dates at self.freq frequency
        """
        if self.freq == '1min':
            return int((end - start).total_seconds() / 60)
        elif self.freq == '1d':
            return (end - start).days
        elif self.freq == 'M':
            return (end.year - start.year) * 12 + (end.month - start.month)
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")
            
    def add_horizon_lags(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add lagged versions of specified columns for each horizon
        
        Args:
            df: DataFrame to add lags to
            cols: List of column names to create lags for
            
        Returns:
            DataFrame with added lag columns
        """
        for col in cols:
            for horizon_name, steps in self.horizons.items():
                df[f"{col}_lag_{horizon_name}"] = df[col].shift(steps)
        return df

    def load_data(self) -> None:
        """Load T1 merged data and sentiment analysis data"""
        logger.info("Loading data...")
        self.t1_merged = pd.read_csv(self.data_dir / "T1_merged.csv")
        self.sentiment_data = pd.read_csv(self.data_dir / "raw_sentiment_results.csv")
        self.sentiment_data = self.sentiment_data.rename(columns={'date': 'datetime'})
        self.stock_mapping = pd.read_csv(self.data_dir / "reddit/stock_index.csv")
        
        # Merge sentiment data with stock mapping
        self.sentiment_data = self.sentiment_data.merge(self.stock_mapping[['id', 'stock_symbol']], left_on='post_id', right_on='id', how='left')
        self.sentiment_data = self.sentiment_data.rename(columns={'datetime': 'timestamp'})
        
        # Convert datetime columns and ensure proper timezone handling
        self.t1_merged['timestamp'] = pd.to_datetime(self.t1_merged['timestamp'], utc=True)
        self.sentiment_data['timestamp'] = pd.to_datetime(self.sentiment_data['timestamp'], utc=True)
        
        # —— ALIGN ALL SENTIMENT TIMES TO THE MINUTE GRID ——
        # Convert to naive datetime (remove timezone info) and floor to nearest minute
        self.sentiment_data['timestamp'] = (
            self.sentiment_data['timestamp']
            .dt.tz_localize(None)
            .dt.floor('T')
        )
        
        # Create a continuous time index from 2018-01-01 to 2020-12-31 with 1-minute frequency
        start_date = pd.Timestamp('2018-01-01')
        end_date = pd.Timestamp('2020-12-31 23:59:00')
        continuous_index = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # Create a pivot table of sentiment scores by stock
        sentiment_pivot = self.sentiment_data.pivot_table(
            index='timestamp',
            columns='stock_symbol',
            values='sentiment_score',
            aggfunc='mean'
        )
        
        # Reindex to continuous time series
        sentiment_pivot = sentiment_pivot.reindex(continuous_index)
        
        # Define rolling windows in minutes
        windows = {
            '30min': 30,
            '2h': 120,
            '12h': 720,
            '1d': 1440,
            '1w': 10080  # 7 days * 24 hours * 60 minutes
        }
        
        # Create a new DataFrame to store only the rolling window features
        rolling_features = pd.DataFrame(index=continuous_index)
        
        # Compute backward-looking rolling means for each stock and window
        for stock in sentiment_pivot.columns:
            # Shift the data by 1 to ensure we don't include current observation
            shifted_data = sentiment_pivot[stock].shift(1)
            
            for window_name, window_size in windows.items():
                # Calculate the count first
                count_col = f'{stock}_sentiment_{window_name}_count'
                rolling_features[count_col] = shifted_data.rolling(
                    window=window_size,
                    min_periods=1,
                    closed='left'
                ).count()
                
                # Calculate the mean, but only where count > 0
                mean_col = f'{stock}_sentiment_{window_name}'
                rolling_features[mean_col] = shifted_data.rolling(
                    window=window_size,
                    min_periods=1,
                    closed='left'
                ).mean()
                
                # Set sentiment to NaN where count is 0
                rolling_features.loc[rolling_features[count_col] == 0, mean_col] = np.nan
        
        # Forward fill missing values for counts only
        count_cols = [col for col in rolling_features.columns if 'count' in col]
        rolling_features[count_cols] = rolling_features[count_cols].ffill()
        
        # Convert T1_merged timestamp to naive datetime to match sentiment_pivot
        self.t1_merged['timestamp'] = self.t1_merged['timestamp'].dt.tz_localize(None)
        
        # Merge with T1_merged data using only the rolling features
        self.t1_merged = self.t1_merged.merge(
            rolling_features,
            left_on='timestamp',
            right_index=True,
            how='left'
        )
        
        # Fill count columns with 0 (no observations)
        count_cols = [col for col in self.t1_merged.columns if 'count' in col]
        self.t1_merged[count_cols] = self.t1_merged[count_cols].fillna(0)
        
        # Fill sentiment columns with NaN where count is 0
        for stock in sentiment_pivot.columns:
            for window_name in windows.keys():
                count_col = f'{stock}_sentiment_{window_name}_count'
                mean_col = f'{stock}_sentiment_{window_name}'
                self.t1_merged.loc[self.t1_merged[count_col] == 0, mean_col] = np.nan
        
        logger.info(f"Successfully merged sentiment data with rolling windows. New columns: {list(rolling_features.columns)}")
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Use base lag of 1 for technical indicators
        shifted_close = df['Close'].shift(1)
        
        # Simple Moving Averages
        df['SMA_3'] = shifted_close.rolling(window=3, min_periods=1, closed='left').mean()
        df['SMA_6'] = shifted_close.rolling(window=6, min_periods=1, closed='left').mean()
        
        # Exponential Moving Averages
        df['EMA_3'] = shifted_close.ewm(span=3, adjust=False).mean()
        df['EMA_6'] = shifted_close.ewm(span=6, adjust=False).mean()
        
        # RSI
        delta = shifted_close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1, closed='left').mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1, closed='left').mean()
        # Add small epsilon to avoid division by zero
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = shifted_close.ewm(span=12, adjust=False).mean()
        exp2 = shifted_close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
        
    def calculate_returns_and_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns and momentum features"""
        # Shift data by 1 to ensure no lookahead bias
        shifted_close = df['Close'].shift(1)
        
        # Log returns with zero handling
        price_ratio = shifted_close / shifted_close.shift(1)
        # Replace zeros and negative values with a small positive number
        price_ratio = price_ratio.replace([0, np.inf, -np.inf], np.nan)
        price_ratio = price_ratio.ffill().fillna(1.0)  # Fill with 1.0 for first value
        # Add small epsilon to avoid zero values
        price_ratio = price_ratio + 1e-10
        df['log_return'] = np.log(price_ratio)
        
        # Multi-horizon returns
        for n in [1, 2, 4]:
            # Handle division by zero in pct_change
            returns = shifted_close.pct_change(n)
            returns = returns.replace([np.inf, -np.inf], np.nan)
            returns = returns.fillna(0)  # Fill with 0 for first values
            df[f'return_{n}block'] = returns
            
        # Momentum
        for n in [1, 2, 4]:
            df[f'momentum_{n}block'] = shifted_close - shifted_close.shift(n)
            
        return df
        
    def calculate_volatility_and_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility and volume features"""
        # Shift data by 1 to ensure no lookahead bias
        shifted_close = df['Close'].shift(1)
        shifted_volume = df['Volume'].shift(1)
        
        # Realized volatility with zero handling
        price_ratio = shifted_close / shifted_close.shift(1)
        price_ratio = price_ratio.replace([0, np.inf, -np.inf], np.nan)
        price_ratio = price_ratio.ffill().fillna(1.0)
        # Add small epsilon to avoid zero values
        price_ratio = price_ratio + 1e-10
        log_returns = np.log(price_ratio)
        
        for n in [3, 6, 12]:
            df[f'realized_vol_{n}block'] = log_returns.rolling(
                window=n, min_periods=1, closed='left'
            ).std()
            
        # Volume features with zero handling
        volume_ratio = shifted_volume / shifted_volume.shift(1)
        volume_ratio = volume_ratio.replace([0, np.inf, -np.inf], np.nan)
        volume_ratio = volume_ratio.ffill().fillna(1.0)  # Fill with 1.0 for first value
        # Add small epsilon to avoid zero values
        volume_ratio = volume_ratio + 1e-10
        df['volume_change'] = volume_ratio - 1
        
        for n in [3, 6]:
            df[f'volume_sum_{n}block'] = shifted_volume.rolling(
                window=n, min_periods=1, closed='left'
            ).sum()
            
        return df
        
    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price"""
        # Shift data by 1 to ensure no lookahead bias
        shifted_close = df['Close'].shift(1)
        shifted_volume = df['Volume'].shift(1)
        
        df['VWAP'] = (shifted_close * shifted_volume).rolling(
            window=3, min_periods=1, closed='left'
        ).sum() / shifted_volume.rolling(
            window=3, min_periods=1, closed='left'
        ).sum()
        
        return df
        
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features"""
        # Cyclical encoding of hour
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        
        # Month and quarter end flags
        df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['timestamp'].dt.is_quarter_end.astype(int)
        
        return df
        
    def calculate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-related features using VIX data"""
        # Shift VIX data by 1 to ensure no lookahead bias
        shifted_vix = df['Close_vix'].shift(1)
        
        # VIX returns
        df['vix_return'] = shifted_vix.pct_change()
        
        # VIX volatility
        df['vix_volatility'] = shifted_vix.rolling(
            window=7,
            min_periods=1,
            closed='left'
        ).std()
        
        # VIX momentum
        df['vix_momentum'] = shifted_vix - shifted_vix.shift(7)
        
        # VIX range (normalized)
        df['vix_range'] = (df['High_vix'] - df['Low_vix']) / df['Close_vix']
        
        return df
        
    def calculate_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate macro-related features using monthly indicators at minute granularity"""
        # Shift all macro data by 1 to ensure no lookahead bias
        shifted_macro = df[['CORESTICKM159SFRBATL', 'NETEXP', 'FEDFUNDS', 
                           'WM1NS', 'WM2NS', 'BSCICP03USM665S', 'CCSA']].shift(1)
        
        # Forward fill macro data to minute granularity
        macro_ffill = shifted_macro.ffill()
        
        # Calculate monthly changes (MoM) using horizon steps
        monthly_steps = self.horizons['train'] // (30*24*60)  # Approximate month in minutes
        monthly_changes = macro_ffill.pct_change(periods=monthly_steps)
        # Handle division by zero
        monthly_changes = monthly_changes.replace([np.inf, -np.inf], np.nan)
        monthly_changes = monthly_changes.fillna(0)
        
        # Calculate 3-month moving averages of the monthly changes
        new_features = {}
        for col in monthly_changes.columns:
            new_features[f'{col}_mom_change'] = monthly_changes[col]
            new_features[f'{col}_3m_ma'] = monthly_changes[col].rolling(
                window=3*monthly_steps,
                min_periods=1,
                closed='left'
            ).mean()
        
        # Calculate composite indicators with safeguards
        # Economic Activity Index (using business confidence and unemployment)
        bci_change = macro_ffill['BSCICP03USM665S'].pct_change(periods=monthly_steps)
        ccs_change = macro_ffill['CCSA'].pct_change(periods=monthly_steps)
        bci_change = bci_change.replace([np.inf, -np.inf], np.nan).fillna(0)
        ccs_change = ccs_change.replace([np.inf, -np.inf], np.nan).fillna(0)
        new_features['economic_activity'] = (bci_change - ccs_change).rolling(
            window=3*monthly_steps,
            min_periods=1,
            closed='left'
        ).mean()
        
        # Monetary Conditions Index (using Fed Funds and M2)
        fed_change = macro_ffill['FEDFUNDS'].pct_change(periods=monthly_steps)
        m2_change = macro_ffill['WM2NS'].pct_change(periods=monthly_steps)
        fed_change = fed_change.replace([np.inf, -np.inf], np.nan).fillna(0)
        m2_change = m2_change.replace([np.inf, -np.inf], np.nan).fillna(0)
        new_features['monetary_conditions'] = (fed_change + m2_change).rolling(
            window=3*monthly_steps,
            min_periods=1,
            closed='left'
        ).mean()
        
        # Inflation Expectations (using sticky CPI and M2)
        cpi_change = macro_ffill['CORESTICKM159SFRBATL'].pct_change(periods=monthly_steps)
        cpi_change = cpi_change.replace([np.inf, -np.inf], np.nan).fillna(0)
        new_features['inflation_expectations'] = (cpi_change + m2_change).rolling(
            window=3*monthly_steps,
            min_periods=1,
            closed='left'
        ).mean()
        
        # Convert dictionary to DataFrame and concatenate with original
        new_features_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_features_df], axis=1)
        
    def calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate focused sentiment features for AAPL prediction
        
        This method creates aggregated sentiment features while handling:
        1. Multicolinearity through feature selection and aggregation
        2. NaN values through proper imputation
        3. Focus on AAPL-specific features
        4. Cross-stock sentiment relationships
        
        Args:
            df: DataFrame containing raw sentiment features
            
        Returns:
            DataFrame with engineered sentiment features
        """
        logger.info("Calculating sentiment features...")
        
        # Define time windows for aggregation
        windows = {
            '30min': 30,
            '2h': 120,
            '12h': 720,
            '1d': 1440,
            '1w': 10080
        }
        
        # Initialize new features dictionary
        new_features = {}
        
        # 1. AAPL-specific sentiment features
        for window_name, window_size in windows.items():
            # Raw sentiment
            sentiment_col = f'aapl_sentiment_{window_name}'
            count_col = f'aapl_sentiment_{window_name}_count'
            
            # Calculate weighted sentiment (weighted by count)
            new_features[f'aapl_weighted_sentiment_{window_name}'] = (
                df[sentiment_col] * df[count_col]
            ).rolling(
                window=window_size,
                min_periods=1,
                closed='left'
            ).sum() / df[count_col].rolling(
                window=window_size,
                min_periods=1,
                closed='left'
            ).sum()
            
            # Calculate sentiment volatility
            new_features[f'aapl_sentiment_vol_{window_name}'] = df[sentiment_col].rolling(
                window=window_size,
                min_periods=1,
                closed='left'
            ).std()
            
            # Calculate sentiment momentum (change in sentiment)
            new_features[f'aapl_sentiment_momentum_{window_name}'] = df[sentiment_col].diff(
                periods=window_size
            )
        
        # 2. Cross-stock sentiment features
        related_stocks = ['msft', 'nvda', 'tsla']  # Tech sector peers
        for stock in related_stocks:
            for window_name in windows.keys():
                # Calculate correlation with AAPL sentiment
                sentiment_col = f'{stock}_sentiment_{window_name}'
                aapl_sentiment = f'aapl_sentiment_{window_name}'
                
                new_features[f'{stock}_aapl_sentiment_corr_{window_name}'] = (
                    df[sentiment_col].rolling(
                        window=windows[window_name],
                        min_periods=1,
                        closed='left'
                    ).corr(df[aapl_sentiment])
                )
                
                # Calculate sentiment spread
                new_features[f'{stock}_aapl_sentiment_spread_{window_name}'] = (
                    df[sentiment_col] - df[aapl_sentiment]
                )
        
        # 3. Market-wide sentiment features
        all_stocks = ['aapl', 'msft', 'nvda', 'tsla', 'gme', 'mcd', 'nflx']
        for window_name in windows.keys():
            # Calculate market sentiment index
            market_sentiment = pd.DataFrame()
            for stock in all_stocks:
                sentiment_col = f'{stock}_sentiment_{window_name}'
                count_col = f'{stock}_sentiment_{window_name}_count'
                if sentiment_col in df.columns:
                    market_sentiment[stock] = df[sentiment_col] * df[count_col]
            
            new_features[f'market_sentiment_index_{window_name}'] = market_sentiment.mean(axis=1)
            
            # Calculate sentiment dispersion
            new_features[f'sentiment_dispersion_{window_name}'] = market_sentiment.std(axis=1)
        
        # 4. Sentiment regime features
        for window_name in windows.keys():
            # Calculate sentiment regime using K-means
            sentiment_cols = [
                f'aapl_sentiment_{window_name}',
                f'aapl_sentiment_vol_{window_name}',
                f'market_sentiment_index_{window_name}'
            ]
            
            # Only calculate if all required columns exist
            if all(col in df.columns for col in sentiment_cols):
                kmeans = KMeans(n_clusters=3, random_state=42)
                regime = kmeans.fit_predict(df[sentiment_cols])
                new_features[f'sentiment_regime_{window_name}'] = regime
        
        # Convert dictionary to DataFrame
        new_features_df = pd.DataFrame(new_features, index=df.index)
        
        # Handle NaN values
        # Forward fill for short-term features
        short_term_cols = [col for col in new_features_df.columns if '30min' in col or '2h' in col]
        new_features_df[short_term_cols] = new_features_df[short_term_cols].ffill()
        
        # Backward fill for long-term features
        long_term_cols = [col for col in new_features_df.columns if '1d' in col or '1w' in col]
        new_features_df[long_term_cols] = new_features_df[long_term_cols].bfill()
        
        # Fill remaining NaN with 0 (neutral sentiment)
        new_features_df = new_features_df.fillna(0)
        
        # Drop highly correlated features
        corr_matrix = new_features_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        new_features_df = new_features_df.drop(columns=to_drop)
        
        logger.info(f"Created {len(new_features_df.columns)} sentiment features")
        return pd.concat([df, new_features_df], axis=1)
        
    def calculate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate meaningful interaction features between different feature categories
        
        Creates interactions between:
        1. Sentiment and price movements
        2. Sentiment and market conditions
        3. Sentiment and technical indicators
        4. Sentiment and macro indicators
        5. Market conditions and technical indicators
        """
        logger.info("Calculating interaction features...")
        
        # Initialize new features dictionary
        new_features = {}
        
        # 1. Sentiment-Price Interactions
        # Use weighted sentiment as it's more robust
        for window in ['30min', '2h', '12h', '1d', '1w']:
            sentiment_col = f'aapl_weighted_sentiment_{window}'
            # Interaction with returns
            new_features[f'sentiment_return_{window}'] = df[sentiment_col] * df['log_return']
            # Interaction with volatility
            new_features[f'sentiment_vol_{window}'] = df[sentiment_col] * df['realized_vol_3block']
            # Interaction with volume
            new_features[f'sentiment_volume_{window}'] = df[sentiment_col] * df['volume_change']
        
        # 2. Sentiment-Market Condition Interactions
        for window in ['30min', '2h', '12h']:
            sentiment_col = f'aapl_weighted_sentiment_{window}'
            # Interaction with VIX
            new_features[f'sentiment_vix_{window}'] = df[sentiment_col] * df['vix_volatility']
            # Interaction with market sentiment
            new_features[f'sentiment_market_{window}'] = (
                df[sentiment_col] * df[f'market_sentiment_index_{window}']
            )
        
        # 3. Sentiment-Technical Indicator Interactions
        for window in ['30min', '2h']:
            sentiment_col = f'aapl_weighted_sentiment_{window}'
            # Interaction with RSI
            new_features[f'sentiment_rsi_{window}'] = df[sentiment_col] * df['RSI_14']
            # Interaction with MACD
            new_features[f'sentiment_macd_{window}'] = df[sentiment_col] * df['MACD']
            # Interaction with momentum
            new_features[f'sentiment_momentum_{window}'] = (
                df[sentiment_col] * df['momentum_1block']
            )
        
        # 4. Sentiment-Macro Interactions
        for window in ['1d', '1w']:
            sentiment_col = f'aapl_weighted_sentiment_{window}'
            # Interaction with economic activity
            new_features[f'sentiment_econ_{window}'] = (
                df[sentiment_col] * df['economic_activity']
            )
            # Interaction with monetary conditions
            new_features[f'sentiment_monetary_{window}'] = (
                df[sentiment_col] * df['monetary_conditions']
            )
            # Interaction with inflation expectations
            new_features[f'sentiment_inflation_{window}'] = (
                df[sentiment_col] * df['inflation_expectations']
            )
        
        # 5. Market Condition-Technical Indicator Interactions
        # VIX-Volatility interaction
        new_features['vix_vol_interaction'] = df['vix_volatility'] * df['realized_vol_3block']
        # VIX-Momentum interaction
        new_features['vix_momentum_interaction'] = df['vix_volatility'] * df['momentum_1block']
        # VIX-RSI interaction
        new_features['vix_rsi_interaction'] = df['vix_volatility'] * df['RSI_14']
        
        # 6. Cross-Stock Sentiment Interactions
        for window in ['30min', '2h', '12h']:
            # Tech sector sentiment interaction
            tech_sentiment = (
                df[f'msft_sentiment_{window}'] + 
                df[f'nvda_sentiment_{window}'] + 
                df[f'tsla_sentiment_{window}']
            ) / 3
            new_features[f'tech_sector_sentiment_{window}'] = (
                df[f'aapl_weighted_sentiment_{window}'] * tech_sentiment
            )
        
        # Convert dictionary to DataFrame
        new_features_df = pd.DataFrame(new_features, index=df.index)
        
        # Handle NaN values
        new_features_df = new_features_df.fillna(0)
        
        # Drop highly correlated features
        corr_matrix = new_features_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        new_features_df = new_features_df.drop(columns=to_drop)
        
        logger.info(f"Created {len(new_features_df.columns)} interaction features")
        return pd.concat([df, new_features_df], axis=1)
        
    def apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA to technical indicators"""
        tech_cols = ['SMA_3', 'SMA_6', 'EMA_3', 'EMA_6', 'RSI_14', 'MACD', 'MACD_signal']
        pca = PCA(n_components=3)
        tech_pca = pca.fit_transform(df[tech_cols])
        df['tech_pc1'] = tech_pca[:, 0]
        df['tech_pc2'] = tech_pca[:, 1]
        df['tech_pc3'] = tech_pca[:, 2]
        
        return df
        
    def apply_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply K-means clustering to sentiment and volatility features"""
        cluster_cols = ['mean_sentiment', 'sentiment_volatility', 'realized_vol_3block']
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['regime'] = kmeans.fit_predict(df[cluster_cols])
        
        return df
        
    def clean_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform features"""
        
        # Define columns to preserve
        preserve_columns = ['High', 'Low', 'Open', 'Close', 'Adj_Close', 'timestamp']
        
        # Winsorization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in preserve_columns]
        
        for col in numeric_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)
            
        # Replace any remaining inf/-inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        df = df.fillna(df.median())
            
        # Scaling
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df
        
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features and reduce multicolinearity
        
        This method:
        1. Logs the initial shape and feature count
        2. Removes constant features
        3. Removes highly correlated features
        4. Removes features with high NaN ratio
        5. Logs the final shape and feature count
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Cleaned DataFrame with reduced multicolinearity
        """
        logger.info("Starting feature cleaning...")
        initial_shape = df.shape
        logger.info(f"Initial shape: {initial_shape}")
        logger.info(f"Initial number of features: {initial_shape[1]}")
        
        # Define columns to preserve
        preserve_columns = ['High', 'Low', 'Open', 'Close', 'Adj_Close', 'timestamp']
        
        # 1. Remove constant features (excluding preserved columns)
        constant_features = [
            col for col in df.columns 
            if col not in preserve_columns and df[col].nunique() == 1
        ]
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features: {constant_features}")
            df = df.drop(columns=constant_features)
        
        # 2. Remove features with high NaN ratio (>50%)
        nan_ratio = df.isna().sum() / len(df)
        high_nan_features = [
            col for col in nan_ratio[nan_ratio > 0.5].index.tolist()
            if col not in preserve_columns
        ]
        if high_nan_features:
            logger.info(f"Removing {len(high_nan_features)} features with high NaN ratio: {high_nan_features}")
            df = df.drop(columns=high_nan_features)
        
        # 3. Remove highly correlated features
        # Calculate correlation matrix (excluding datetime columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than 0.95 (excluding preserved columns)
        to_drop = [
            column for column in upper.columns 
            if column not in preserve_columns and any(upper[column] > 0.95)
        ]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
            df = df.drop(columns=to_drop)
        
        # 4. Remove features with low variance (excluding preserved columns)
        # Calculate variance for each numeric feature
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()
        low_var_features = [
            col for col in variances[variances < 1e-10].index.tolist()
            if col not in preserve_columns
        ]
        if low_var_features:
            logger.info(f"Removing {len(low_var_features)} features with low variance: {low_var_features}")
            df = df.drop(columns=low_var_features)
        
        # 5. Fill remaining NaN values
        # For numerical features, use median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Log final shape
        final_shape = df.shape
        logger.info(f"Final shape: {final_shape}")
        logger.info(f"Final number of features: {final_shape[1]}")
        logger.info(f"Removed {initial_shape[1] - final_shape[1]} features in total")
        
        return df

    def engineer_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Main method to engineer all features
        
        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames
        """
        logger.info("Starting feature engineering...")
        
        # Load data
        self.load_data()
        df = self.t1_merged
        
        logger.info(f"Initial columns: {df.columns}")
        
        # Calculate all feature groups
        df = self.calculate_technical_indicators(df)
        df = self.calculate_returns_and_momentum(df)
        df = self.calculate_volatility_and_volume(df)
        df = self.calculate_vwap(df)
        df = self.add_calendar_features(df)
        df = self.calculate_market_features(df)
        df = self.calculate_macro_features(df)
        df = self.calculate_sentiment_features(df)
        df = self.calculate_interaction_features(df)
        
        # Clean features and reduce multicolinearity
        df = self.clean_features(df)
        
        # Final cleaning and transformation
        df = self.clean_and_transform(df)
        
        # Set timestamp as index for date-based slicing
        df = df.set_index('timestamp')
        
        # Split into train/val/test sets
        train_df = df.loc[(df.index >= self.train_start) & (df.index <= self.train_end)].copy()
        val_df = df.loc[(df.index >= self.val_start) & (df.index <= self.val_end)].copy()
        test_df = df.loc[(df.index >= self.test_start) & (df.index <= self.test_end)].copy()
        
        return train_df, val_df, test_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature Engineering for Trading Strategy')
    parser.add_argument('--train-period', type=str, required=True,
                      help='Training period in format "start_date,end_date"')
    parser.add_argument('--val-period', type=str, required=True,
                      help='Validation period in format "start_date,end_date"')
    parser.add_argument('--test-period', type=str, required=True,
                      help='Test period in format "start_date,end_date"')
    parser.add_argument('--freq', type=str, default='1min',
                      help='Data frequency (default: 1min)')
    parser.add_argument('--output-dir', type=str, default='data',
                      help='Output directory for saving datasets')
    parser.add_argument('--export-individual', action='store_true',
                      help='Export train, val, and test datasets individually')
    
    args = parser.parse_args()
    
    # Parse date periods
    train_start, train_end = args.train_period.split(',')
    val_start, val_end = args.val_period.split(',')
    test_start, test_end = args.test_period.split(',')
    
    # Initialize feature engineer
    engineer = FeatureEngineer(
        train_period=(train_start, train_end),
        val_period=(val_start, val_end),
        test_period=(test_start, test_end),
        freq=args.freq
    )
    
    # Engineer features
    train_df, val_df, test_df = engineer.engineer_features()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.export_individual:
        # Export individual datasets
        train_df.to_csv(output_dir / 'train_features.csv')
        val_df.to_csv(output_dir / 'val_features.csv')
        test_df.to_csv(output_dir / 'test_features.csv')
        logger.info("Exported individual train, validation, and test datasets")
    else:
        # Export combined dataset
        combined_df = pd.concat([train_df, val_df, test_df])
        combined_df.to_csv(output_dir / 'T2_engineered_features.csv')
        logger.info("Exported combined dataset")
    
    logger.info("Feature engineering completed successfully!")

if __name__ == "__main__":
    main() 