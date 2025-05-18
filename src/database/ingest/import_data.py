import pandas as pd
from supabase import create_client
import os
from tqdm import tqdm
import numpy as np
import time

# Supabase configuration
SUPABASE_URL = "https://wlwnbvqkmmnmlyhvygrv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indsd25idnFrbW1ubWx5aHZ5Z3J2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ3MDk1MzYsImV4cCI6MjA2MDI4NTUzNn0.BPlGrxvMpMnP3PcAUtlf8W1PncqFOCjD_1AFZENyHj8"

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean_record(record):
    """Clean a single record to ensure JSON compatibility"""
    cleaned = {}
    for key, value in record.items():
        # Handle date format
        if key == 'Date':
            cleaned[key] = str(value)
            continue
            
        # Replace NaN/infinite values with None
        if pd.isna(value) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            cleaned[key] = None
        # Convert numpy int64/float64 to native Python types
        elif isinstance(value, (np.int64, np.int32)):
            cleaned[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            # Round to 2 decimal places for price columns
            if key in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                cleaned[key] = round(float(value), 2)
            # Convert Volume to integer, handling any decimal points
            elif key == 'Volume':
                # Round to nearest integer and convert to int
                cleaned[key] = int(round(float(value)))
            else:
                cleaned[key] = float(value)
        else:
            # Keep other values as is
            cleaned[key] = value
    return cleaned

def import_posts():
    # Read the CSV in chunks due to its large size
    chunk_size = 1000  # Adjust based on your memory constraints
    
    # Read the CSV with proper parsing
    chunks = pd.read_csv('aapl_1min_data_2018_2020.csv', 
                        chunksize=chunk_size,
                        parse_dates=['Date'],
                        dtype={'Volume': float})
    
    total_rows = sum(1 for _ in open('aapl_1min_data_2018_2020.csv')) - 1  # -1 for header
    progress_bar = tqdm(total=total_rows, desc="Importing posts")
    
    for chunk in chunks:
        # Convert chunk to dictionary format and clean each record
        records = [clean_record(record) for record in chunk.to_dict('records')]
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Insert chunk into Supabase
                data, count = supabase.table('aapl').insert(records).execute()
                progress_bar.update(len(records))
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Error inserting chunk after {max_retries} attempts: {str(e)}")
                    print(f"Sample problematic record: {records[0]}")
                    print("\nFirst few records in chunk:")
                    for i, record in enumerate(records[:3]):
                        print(f"Record {i}: {record}")
                else:
                    print(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
                    time.sleep(2 ** retry_count)  # Exponential backoff
    
    progress_bar.close()

if __name__ == "__main__":
    print("Starting data import...")
    
    # Import posts first (larger dataset)
    print("Importing posts...")
    import_posts()
    
    print("\nImport completed!") 