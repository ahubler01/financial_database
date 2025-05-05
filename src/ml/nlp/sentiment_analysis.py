import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
from datetime import datetime
from tqdm import tqdm
import pytz
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Data class to store sentiment analysis results"""
    stock_symbol: str
    date: datetime
    sentiment_numeric: float
    sentiment_score: float
    post_id: str
    subreddit: str
    title: str
    selftext: str
    author: str

class RedditSentimentAnalyzer:
    """Class to handle Reddit sentiment analysis for stocks"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", year_range: Tuple[int, int] = (2018, 2020)):
        """Initialize the sentiment analyzer with a specific model and year range
        
        Args:
            model_name: Name of the sentiment analysis model to use
            year_range: Tuple of (start_year, end_year) inclusive
        """
        self.model_name = model_name
        self.year_range = year_range
        self.finbert = None
        self.posts_df = None
        self.index_df = None
        self.subscribers_df = None
        self.merged_df = None
        self.results = []
        
    def load_data(self, posts_path: str, index_path: str, subscribers_path: str) -> None:
        """Load and prepare the data"""
        logger.info("Loading data...")
        self.posts_df = pd.read_csv(posts_path)
        self.index_df = pd.read_csv(index_path)
        self.subscribers_df = pd.read_csv(subscribers_path)
        
        # Merge dataframes
        self.merged_df = self.posts_df.merge(self.index_df, on=['id', 'created_utc'])
        self.merged_df['created_utc'] = pd.to_datetime(self.merged_df['created_utc'], unit='s')
        
        # Convert UTC to ET
        et = pytz.timezone('America/New_York')
        self.merged_df['created_et'] = self.merged_df['created_utc'].dt.tz_localize('UTC').dt.tz_convert(et)
        
        # Filter by year range
        start_year, end_year = self.year_range
        self.merged_df = self.merged_df[
            (self.merged_df['created_et'].dt.year >= start_year) & 
            (self.merged_df['created_et'].dt.year <= end_year)
        ]
        logger.info(f"Filtered data to years {start_year}-{end_year}")
        
    def initialize_model(self) -> None:
        """Initialize the FinBERT model"""
        logger.info("Initializing FinBERT model...")
        self.finbert = pipeline("text-classification", model=self.model_name)
        
    def get_sentiment(self, text: str) -> tuple:
        """Get sentiment score for a given text"""
        try:
            result = self.finbert(text[:512])[0]  # Limit text length to 512 tokens
            if result['label'] == 'positive':
                return 1, result['score']
            elif result['label'] == 'negative':
                return -1, result['score']
            else:
                return 0, result['score']
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return 0, 0.5  # Return neutral sentiment on error
            
    def analyze_stock(self, stock_symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Analyze sentiment for a specific stock"""
        logger.info(f"Analyzing sentiment for {stock_symbol}...")
        
        # Filter data for specific stock
        df = self.merged_df[self.merged_df['stock_symbol'] == stock_symbol.lower()].copy()
        
        if limit:
            df = df.head(limit)
            
        # Combine title and selftext
        df['combined'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
        
        # Apply sentiment analysis
        tqdm.pandas(desc=f"Analyzing sentiment for {stock_symbol}")
        df['sentiment_numeric'], df['sentiment_score'] = zip(*df['combined'].progress_apply(self.get_sentiment))
        
        # Store results
        for _, row in df.iterrows():
            self.results.append(SentimentResult(
                stock_symbol=stock_symbol,
                date=row['created_et'],
                sentiment_numeric=row['sentiment_numeric'],
                sentiment_score=row['sentiment_score'],
                post_id=row['id'],
                subreddit=row['subreddit'],
                title=row['title'],
                selftext=row['selftext'],
                author=row['author']
            ))
            
        return df
        
    def analyze_multiple_stocks(self, stock_symbols: List[str], limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Analyze sentiment for multiple stocks"""
        results = {}
        for symbol in stock_symbols:
            results[symbol] = self.analyze_stock(symbol, limit)
        return results
        
    def get_daily_sentiment(self, stock_symbol: str) -> pd.DataFrame:
        """Get daily sentiment averages for a stock"""
        df = pd.DataFrame([vars(r) for r in self.results if r.stock_symbol == stock_symbol])
        if df.empty:
            return pd.DataFrame()
            
        df['date'] = df['date'].dt.date
        return df.groupby('date')['sentiment_numeric'].mean().reset_index()
        
    def plot_results(self, stock_symbol: str, save_path: str = None) -> None:
        """Plot sentiment analysis results"""
        df = pd.DataFrame([vars(r) for r in self.results if r.stock_symbol == stock_symbol])
        if df.empty:
            logger.warning(f"No data found for {stock_symbol}")
            return
            
        daily_sentiment = self.get_daily_sentiment(stock_symbol)
        
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Daily sentiment trend
        plt.subplot(2, 1, 1)
        plt.plot(daily_sentiment['date'], daily_sentiment['sentiment_numeric'], marker='o')
        plt.title(f'Daily Sentiment Trend for {stock_symbol.upper()} ({self.year_range[0]}-{self.year_range[1]})')
        plt.xlabel('Date (ET)')
        plt.ylabel('Average Sentiment (-1 to 1)')
        plt.grid(True)
        
        # Plot 2: Sentiment distribution
        plt.subplot(2, 1, 2)
        sns.histplot(data=df, x='sentiment_numeric', bins=30)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Score (-1 to 1)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def get_statistics(self, stock_symbol: str) -> Dict[str, float]:
        """Get sentiment statistics for a stock"""
        df = pd.DataFrame([vars(r) for r in self.results if r.stock_symbol == stock_symbol])
        if df.empty:
            return {}
            
        return {
            'total_posts': len(df),
            'mean_sentiment': df['sentiment_numeric'].mean(),
            'median_sentiment': df['sentiment_numeric'].median(),
            'std_sentiment': df['sentiment_numeric'].std()
        }

    def export_sentiment_results(self, output_path: str) -> None:
        """Export sentiment analysis results to CSV
        
        Args:
            output_path: Path where the CSV file will be saved
        """
        if not self.results:
            logger.warning("No results to export")
            return
            
        # Convert results to DataFrame
        df = pd.DataFrame([vars(r) for r in self.results])
        
        # Map numeric sentiment to labels
        df['sentiment_label'] = df['sentiment_numeric'].map({
            1: 'positive',
            -1: 'negative',
            0: 'neutral'
        })
        
        # Select and rename columns
        export_df = df[[
            'date',
            'post_id',
            'subreddit',
            'author',
            'sentiment_label',
            'sentiment_score'
        ]]
        
        # Export to CSV
        export_df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")

def main():
    # Initialize analyzer with custom year range
    analyzer = RedditSentimentAnalyzer(year_range=(2018, 2020))
    
    # Load data
    analyzer.load_data(
        posts_path='data/reddit/posts.csv',
        index_path='data/reddit/stock_index.csv',
        subscribers_path='data/reddit/subreddit_subscribers.csv'
    )
    
    # Initialize model
    analyzer.initialize_model()
    
    # Get unique stocks
    unique_stocks = analyzer.merged_df['stock_symbol'].unique()
    logger.info(f"Found {len(unique_stocks)} unique stocks")
    
    # Analyze all stocks
    results = analyzer.analyze_multiple_stocks(unique_stocks)
    
    # Export results to CSV
    analyzer.export_sentiment_results('data/raw_sentiment_results.csv')
    
    # Process and visualize results for each stock
    for stock in unique_stocks:
        logger.info(f"\nProcessing {stock.upper()}")
        
        # Get statistics
        stats = analyzer.get_statistics(stock)
        logger.info(f"Statistics for {stock.upper()}:")
        for key, value in stats.items():
            logger.info(f"{key}: {value:.3f}")
            
        # Plot results
        analyzer.plot_results(stock, save_path=f'sentiment_analysis_{stock}.png')

if __name__ == "__main__":
    main()



