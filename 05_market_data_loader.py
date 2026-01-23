"""
Project: Quantifying the Narrative
Module: 05_market_data_loader.py
Description: 
    Aggregates Market Data.
    - Yields: FRED API (DGS2, DGS10)
    - Equities: WRDS CRSP (dsp500 -> sprtrn)
    Calculates the 'Same-Day Market Reaction'.
    OUTPUT: 'data/modified_data/fomc_full_dataset.csv'
"""

import pandas as pd
import wrds
from fredapi import Fred
import os
import numpy as np

class MarketDataEngine:
    def __init__(self):
        # --- CREDENTIALS ---
        self.fred_api_key = '23dd8644a8456a82f3dc0e07c51e2a9b' 
        
        # --- PATHS ---
        self.base_dir = "data/modified_data"
        self.sentiment_file = os.path.join(self.base_dir, "fomc_sentiment.csv")
        self.drift_file = os.path.join(self.base_dir, "fomc_drift.csv")
        self.output_file = os.path.join(self.base_dir, "fomc_full_dataset.csv")

    def get_yields(self):
        """
        Fetches US Treasury Yield Levels from FRED.
        """
        print("Fetching Treasury Yields from FRED...")
        fred = Fred(api_key=self.fred_api_key)
        
        # DGS2 & DGS10 (Daily Closing Yields)
        df_2y = fred.get_series('DGS2', observation_start='2000-01-01')
        df_10y = fred.get_series('DGS10', observation_start='2000-01-01')
        
        df = pd.DataFrame({'US2Y': df_2y, 'US10Y': df_10y})
        df.index.name = 'Date'
        return df

    def get_spx_wrds(self):
        """
        Fetches S&P 500 Total Return from WRDS (CRSP).
        Table: crsp.dsp500
        Column: sprtrn (Value-Weighted Return including Dividends)
        """
        print("Connecting to WRDS (CRSP)...")
        db = wrds.Connection() # This will prompt for login if not cached
        
        # The Professional Query
        # We grab 'caldt' (Calendar Date) and 'sprtrn' (S&P 500 Return)
        query = """
        SELECT caldt, sprtrn
        FROM crsp.dsp500
        WHERE caldt >= '2000-01-01'
        ORDER BY caldt
        """
        
        print("Executing SQL Query...")
        data = db.raw_sql(query)
        
        # Close connection
        db.close()
        
        # Clean Data
        # WRDS returns dates as 'objects' (strings) or date objects. convert to Timestamp.
        data['Date'] = pd.to_datetime(data['caldt'])
        data = data.set_index('Date')
        
        # Rename column to be consistent
        # Multiply by 100 to convert decimal (0.01) to percentage (1.0)
        data['SP500_Ret'] = data['sprtrn'] * 100
        
        # Drop the original 'caldt' column
        return data[['SP500_Ret']]

    def run(self):
        # 1. Load NLP Data
        if not os.path.exists(self.sentiment_file):
            print("Error: Sentiment data missing.")
            return
            
        df_sent = pd.read_csv(self.sentiment_file)
        df_drift = pd.read_csv(self.drift_file)
        
        # Merge Sentiment and Drift
        df_nlp = pd.merge(df_sent, df_drift[['Date', 'Drift_Score']], on='Date', how='inner')
        df_nlp['Date'] = pd.to_datetime(df_nlp['Date'])
        
        # 2. Get Market Data
        df_yields = self.get_yields() # Index is Datetime
        df_spx = self.get_spx_wrds()  # Index is Datetime
        
        # Merge Yields and SPX (Inner Join ensures matching dates)
        df_market = pd.merge(df_yields, df_spx, left_index=True, right_index=True, how='inner')
        
        # 3. Calculate "Same-Day Reaction"
        
        # BOND LOGIC: FRED gives Levels (4.00%). We need Change (Today - Yesterday).
        # .diff() calculates (Row_T - Row_T-1)
        df_market['US2Y_Change'] = df_market['US2Y'].diff() * 100 # in Basis Points
        df_market['US10Y_Change'] = df_market['US10Y'].diff() * 100 # in Basis Points
        
        # STOCK LOGIC: WRDS 'sprtrn' IS the return for that day.
        # It represents the change from Yesterday Close to Today Close (incl Divs).
        # So we DO NOT use .diff() or .pct_change() on it. It is already the reaction.
        
        # 4. Final Merge (Event Study Alignment)
        # We align the Meeting Date with the Market Date.
        df_final = pd.merge(df_nlp, df_market, left_on='Date', right_index=True, how='inner')
        
        # 5. Save
        df_final.to_csv(self.output_file, index=False)
        
        print("-" * 50)
        print(f"Dataset Built. Saved to: {self.output_file}")
        print(f"Total Aligned Meetings: {len(df_final)}")
        print("First 5 Rows:")
        print(df_final[['Date', 'Sentiment_Score', 'Drift_Score', 'US2Y_Change', 'SP500_Ret']].head())

if __name__ == "__main__":
    engine = MarketDataEngine()
    engine.run()