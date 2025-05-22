import pandas as pd
import numpy as np
from scipy.stats import mstats
from pathlib import Path
from typing import Dict, List
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self):
        # Get the project root directory (two levels up from the current file)
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.stock_dir = self.data_dir / "stock"
        self.macro_dir = self.data_dir / "macro"
        
        # Log the paths for debugging
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Stock directory: {self.stock_dir}")
        logger.info(f"Macro directory: {self.macro_dir}")
        
    def load_vix_data(self) -> pd.DataFrame:
        """Load and process VIX data"""
        logger.info("Loading VIX data...")
        vix_path = self.stock_dir / "vix_2018_2020.csv"
        logger.info(f"Loading VIX data from: {vix_path}")
        vix_df = pd.read_csv(vix_path)
        vix_df.rename(columns={'DATE': 'timestamp', 'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close'}, inplace=True)
        vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'])
        vix_df.set_index('timestamp', inplace=True)
        
        # Forward fill missing values
        vix_df = vix_df.ffill()
        
        # Remove any remaining NaN values
        vix_df = vix_df.dropna()
        
        return vix_df
    
    def load_aapl_data(self) -> pd.DataFrame:
        """Load and process AAPL data"""
        logger.info("Loading AAPL data...")
        aapl_path = self.stock_dir / "aapl_1min_data_2018_2020.csv"
        logger.info(f"Loading AAPL data from: {aapl_path}")
        aapl_df = pd.read_csv(aapl_path)
        aapl_df.rename(columns={'Date': 'timestamp', 'Adj Close': 'Adj_Close'}, inplace=True)
        aapl_df['timestamp'] = pd.to_datetime(aapl_df['timestamp'])
        aapl_df.set_index('timestamp', inplace=True)
        
        # Forward fill missing values
        aapl_df = aapl_df.ffill()
        # Remove any remaining NaN values
        aapl_df = aapl_df.dropna()

        return aapl_df
    
    def load_macro_data(self) -> pd.DataFrame:
        """Load and process all macro data files"""
        logger.info("Loading macro data...")
        
        macro_dfs = []
        for file in self.macro_dir.glob("*.csv"):
            logger.info(f"Loading macro data from: {file}")
            df = pd.read_csv(file)
            # Get everything before _2018_2020
            col_name = file.stem.split('_2018_2020')[0]
            df.rename(columns={'observation_date': 'timestamp', 'value': col_name}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            macro_dfs.append(df)
        
        # Merge all macro data
        macro_df = pd.concat(macro_dfs, axis=1)
        
        # Forward fill missing values
        macro_df = macro_df.ffill()
        
        # Remove any remaining NaN values
        macro_df = macro_df.dropna()
        
        return macro_df
    
    def process_data(self) -> Dict[str, pd.DataFrame]:
        """Main processing pipeline that returns three separate clean tables"""
        # Load and process each data type
        vix_df   = self.load_vix_data()
        aapl_df  = self.load_aapl_data()
        macro_df = self.load_macro_data()
        
        # — PREPARE VIX FOR AS-OF JOIN —
        # index is a business-day date; shift to that date at 16:00
        vix_asof = vix_df.copy()
        vix_asof.index = (
            pd.to_datetime(vix_asof.index)    # date-of-vix
              .normalize()                    # midnight
              + pd.Timedelta(hours=16)        # 16:00
        )
        vix_asof = vix_asof.sort_index()
        
        # — PREPARE MACRO FOR AS-OF JOIN —
        # stamp every release to its date at 09:30
        macro_asof = macro_df.copy()
        macro_asof.index = (
            pd.to_datetime(macro_asof.index)  # actual release timestamp or date
              .normalize()                    # midnight
              + pd.Timedelta(hours=9, minutes=30)
        )
        macro_asof = macro_asof.sort_index()
        
        # — PREPARE AAPL FOR JOIN —
        aapl_merged = aapl_df.copy().sort_index()
        
        logger.info("Merging VIX (as of 18:00) into minute-level AAPL…")
        aapl_merged = pd.merge_asof(
            left        = aapl_merged,
            right       = vix_asof,
            left_index  = True,
            right_index = True,
            direction   = 'backward',
            suffixes    = ('', '_vix')
        )
        
        logger.info("Merging macro releases (effective 09:30) into minute-level AAPL…")
        aapl_merged = pd.merge_asof(
            left        = aapl_merged,
            right       = macro_asof,
            left_index  = True,
            right_index = True,
            direction   = 'backward',
            suffixes    = ('', '_macro')
        )
        
        return {
            'T1_apple':  aapl_df,
            'T1_stock':  vix_df,
            'T1_macro':  macro_df,
            'T1_merged': aapl_merged
        }


if __name__ == "__main__":
    transformer = DataTransformer()
    processed_data = transformer.process_data()
    
    # Save each table separately
    for table_name, df in processed_data.items():
        output_path = transformer.data_dir / f"{table_name}.csv"
        df.to_csv(output_path)
        logger.info(f"Saved {table_name} with shape {df.shape} to {output_path}")
        logger.info(f"Columns in {table_name}: {df.columns.tolist()}")
