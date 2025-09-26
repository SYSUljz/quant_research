import time
import os
import tushare as ts
import pandas as pd
from pymongo import MongoClient, UpdateOne
from datetime import datetime, timedelta
import panda_data
# --- Configuration ---
# IMPORTANT: Replace with your Tushare token.
# You can get a free token from the Tushare website.
# It's recommended to set this as an environment variable for security.
TUSHARE_TOKEN = '94f520e0621fbeaef1471aa3e8c747e67d24898418d3412522f0fa60'
# MongoDB connection details
MONGO_URI = 'mongodb://panda:panda@127.0.0.1:27017/?authSource=admin'
MONGO_DB_NAME = 'panda'
MONGO_COLLECTION_NAME = 'stock_market'

# Stock to be downloaded


def initialize_connections():
    """
    Initializes and returns the Tushare Pro API and MongoDB collection objects.
    """
    # Initialize Tushare Pro API
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        # A simple check to see if the API is working
        pro.trade_cal(exchange='', start_date='20250101', end_date='20250101')
    except Exception as e:
        print(f"Error initializing Tushare API. Please check your token. Error: {e}")
        return None, None

    # Initialize MongoDB connection
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        # Create a unique index to prevent duplicates at the database level
        collection.create_index([("symbol", 1), ("date", 1)], unique=True)
    except Exception as e:
        print(f"Error connecting to MongoDB. Please check your connection string and credentials. Error: {e}")
        return pro, None

    return pro, collection

def get_last_date(collection, ts_code):
    """
    Finds the most recent trade date for a given stock code in the MongoDB collection.
    """
    latest_entry = collection.find_one(
        {"symbol": ts_code},
        sort=[("date", -1)]
    )
    if latest_entry and 'date' in latest_entry:
        print(f"Last recorded date for {ts_code} is {latest_entry['date']}.")
        return latest_entry['date']
    else:
        print(f"No previous data found for {ts_code}. Will fetch data from the default start date.")
        return None

def fetch_stock_data(pro, ts_code, start_date, end_date):
    """
    Fetches daily stock data from Tushare.
    """
    print(f"Fetching data for {ts_code} from {start_date} to {end_date}...")
    time.sleep(0.5)
    try:
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=[
                "ts_code", "trade_date", "open", "high", "low", "close",
                "pre_close", "vol", "amount"
            ]
        )
        return df
    except Exception as e:
        print(f"An error occurred while fetching data from Tushare: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

def process_and_store_data(collection, df, index_component):
    """
    Processes the DataFrame and upserts the data into MongoDB, avoiding duplicates.
    """
    if df.empty:
        print("No new data to store.")
        return 0

    # The data from Tushare comes in descending order. We reverse it to process chronologically.
    df = df.iloc[::-1]

    operations = []
    for _, row in df.iterrows():
        # Calculate limit up/down (assuming a standard 10% limit for non-special stocks)
        # Note: This is a simplification. Real-world limits can vary (e.g., 20% for STAR market, 5% for ST stocks).
        limit_up = round(row['pre_close'] * 1.1, 2)
        limit_down = round(row['pre_close'] * 0.9, 2)

        # Create the document to be inserted/updated
        document = {
            'symbol': row['ts_code'],
            'date': row['trade_date'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'pre_close': row['pre_close'],
            'volume': row['vol'],
            'amount': row['amount'],
            'limit_up': limit_up,
            'limit_down': limit_down,
            'index_component': index_component
        }

        # Prepare an 'upsert' operation.
        # This will update the document if it exists, or insert it if it does not.
        # The filter checks for a document with the same symbol and date.
        operations.append(
            UpdateOne(
                {"symbol": document['symbol'], "date": document['date']},
                {"$set": document},
                upsert=True
            )
        )

    if not operations:
        print("No operations to perform.")
        return 0

    # Execute all operations in a single batch
    try:
        result = collection.bulk_write(operations)
        print(f"Bulk write operation summary:")
        print(f"  - Matched documents: {result.matched_count}")
        print(f"  - Modified documents: {result.modified_count}")
        print(f"  - Upserted documents: {result.upserted_count}")
        return result.upserted_count
    except Exception as e:
        print(f"An error occurred during bulk write to MongoDB: {e}")
        return 0


def main():
    """
    Main function to run the data download and storage process.
    """
    print("--- Starting Stock Data Synchronization ---")
    pro, collection = initialize_connections()

    if not pro or collection is None:
        print("Initialization failed. Exiting.")
        return
    panda_data.init()
    symbols = panda_data.get_all_symbols()
    work_flow=[]
    # 1. Determine the date range
    for stock_code in symbols[0]:
        index_component = panda_data.get_index_component(stock_code)
        last_date_str = get_last_date(collection, stock_code)
        if last_date_str:
            # Start from the day after the last recorded date
            start_dt = datetime.strptime(last_date_str, '%Y%m%d') + timedelta(days=1)
            start_date = start_dt.strftime('%Y%m%d')

        end_date = '20250825'

        # 2. Fetch data from Tushare
        if last_date_str and start_date >= end_date:
            print(f"No new data to fetch for {stock_code}.")
            continue
        work_flow.append(stock_code)
    total_stocks = len(work_flow)
    for i, stock_code in enumerate(work_flow, start=1):
        print(f"\n[{i}/{total_stocks}] Processing stock: {stock_code}")

        # Step 1: Fetch stock data
        print("  Step 1: Fetching stock data...")
        start_time = time.time()
        stock_df = fetch_stock_data(pro, stock_code, start_date, end_date)
        print(f"  Step 1 complete. Time used: {time.time() - start_time:.2f}s")

        # Step 2: Process and store
        if stock_df is not None:
            print("  Step 2: Processing and storing data in MongoDB...")
            start_time = time.time()
            new_records_count = process_and_store_data(collection, stock_df, index_component)
            print(f"  Step 2 complete. Added {new_records_count} records. Time used: {time.time() - start_time:.2f}s")
        else:
            print("  Step 2 skipped (no data).")

    print("\n=== All stocks processed ===")

if __name__ == "__main__":
    if TUSHARE_TOKEN == "YOUR_TUSHARE_TOKEN_HERE":
        print("Please set your TUSHARE_TOKEN in the script before running.")
    else:
        main()
