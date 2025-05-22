import pandas as pd
from supabase import create_client
import os
from tqdm import tqdm
import numpy as np

# Supabase configuration
SUPABASE_URL = ""
SUPABASE_KEY = ""

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean_record(record):
    """Clean a single record to ensure JSON compatibility"""
    for key, value in record.items():
        # Replace NaN/infinite values with None
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            record[key] = None
        # Convert numpy int64/float64 to native Python types
        elif isinstance(value, (np.int64, np.int32)):
            record[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            record[key] = float(value)
    return record

def import_posts():
    # Read the CSV in chunks due to its large size
    chunk_size = 1000  # Adjust based on your memory constraints
    chunks = pd.read_csv('data/posts.csv', chunksize=chunk_size)
    
    total_rows = sum(1 for _ in open('data/posts.csv')) - 1  # -1 for header
    progress_bar = tqdm(total=total_rows, desc="Importing posts")
    
    for chunk in chunks:
        # Convert chunk to dictionary format and clean each record
        records = [clean_record(record) for record in chunk.to_dict('records')]
        
        try:
            # Insert chunk into Supabase
            data, count = supabase.table('posts').insert(records).execute()
            progress_bar.update(len(records))
        except Exception as e:
            print(f"Error inserting chunk: {e}")
            # Print a sample of the problematic records for debugging
            if records:
                print(f"Sample problematic record: {records[0]}")
            continue
    
    progress_bar.close()

if __name__ == "__main__":
    print("Starting data import...")
    
    # Import posts first (larger dataset)
    print("Importing posts...")
    import_posts()
    

    
    print("\nImport completed!") 