import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = "https://wlwnbvqkmmnmlyhvygrv.supabase.co"
key: str = os.getenv("SUPABASE_KEY")  # You'll need to set this in your .env file
supabase: Client = create_client(url, key)

def fetch_aapl_data():
    try:
        # Fetch all data from aapl_data table
        response = supabase.table('posts').select("*").execute()
        
        # Print the data
        print("Successfully fetched data from aapl_data table:")
        print(response.data)
        
        return response.data
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

if __name__ == "__main__":
    fetch_aapl_data() 