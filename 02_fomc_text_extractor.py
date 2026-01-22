"""
Project: Quantifying the Narrative
Module: 02_fomc_text_extractor.py
Description: 
    Reads the link manifest from 'data/raw_data/fomc_links.csv'.
    Visits every URL.
    Extracts policy text using Multi-Strategy Parsing (ID-based + Density Heuristic).
    Saves individual .txt files into 'data/raw_data/statements/'.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
import random
import re

class FOMCTextExtractor:
    def __init__(self):
        # --- PATH CONFIGURATION ---
        self.base_dir = "data/raw_data"
        self.input_csv = os.path.join(self.base_dir, "fomc_links.csv")
        self.output_folder = os.path.join(self.base_dir, "statements")
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Ensure output directory exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"[+] Created directory: {self.output_folder}")

    def clean_text(self, text: str) -> str:
        """
        Sanitizes the text. 
        Replaces multiple newlines/tabs with single spaces.
        """
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_date_from_url(self, url: str, year: int) -> str:
        """
        Parses the URL to find the exact date (YYYY-MM-DD).
        Example URL: .../monetary20081028a.htm -> Returns 2008-10-28
        """
        match = re.search(r'(\d{8})', url)
        if match:
            date_str = match.group(1)
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        else:
            return f"{year}_unknown_date"

    def extract_text_from_url(self, url: str) -> str:
        """
        The Core Logic. Tries 3 strategies to handle HTML Layouts from 2000-2024.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                print(f"  [!] HTTP Error {response.status_code} for {url}")
                return ""
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # --- STEP 1: PRE-CLEANING ---
            # Remove scripts, styles, and navigation elements
            for script in soup(["script", "style", "nav", "footer", "header", "input", "button"]):
                script.decompose()

            # --- STRATEGY A: Known Containers (2006-2024) ---
            golden_ids = ['article', 'leftText', 'content']
            for gid in golden_ids:
                div = soup.find('div', {'id': gid})
                if div:
                    return self.clean_text(div.get_text())
            
            # Check for Bootstrap class (2010-2015 era)
            div = soup.find('div', {'class': 'col-md-8'})
            if div:
                return self.clean_text(div.get_text())

            # --- STRATEGY B: The Density Heuristic (2000-2005) ---
            # Finds the table cell or div with the most text characters.
            candidates = []
            for element in soup.find_all(['td', 'div']):
                text = element.get_text()
                clean = " ".join(text.split())
                
                # Filter navigation bars
                if len(clean) < 200: continue
                candidates.append(clean)
            
            if candidates:
                # Pick the longest text block
                candidates.sort(key=len, reverse=True)
                return self.clean_text(candidates[0])

            # --- STRATEGY C: Fallback ---
            return self.clean_text(soup.get_text())

        except Exception as e:
            print(f"  [!] Exception for {url}: {e}")
            return ""

    def run(self):
        # 1. Validation
        if not os.path.exists(self.input_csv):
            print(f"CRITICAL ERROR: Could not find {self.input_csv}")
            return
            
        df = pd.read_csv(self.input_csv)
        print(f"Loaded Manifest: {len(df)} statements found.")
        print(f"Saving to: {self.output_folder}")
        print("-" * 50)
        
        success_count = 0
        
        # 2. Iteration
        for index, row in df.iterrows():
            year = row['Year']
            url = row['URL']
            
            # Generate Filename
            date_str = self.extract_date_from_url(url, year)
            filename = f"{date_str}.txt"
            
            # Handle rare collision
            if "unknown" in date_str:
                filename = f"{year}_{index}.txt"
                
            filepath = os.path.join(self.output_folder, filename)
            
            # Idempotency Check (Skip if exists)
            if os.path.exists(filepath):
                success_count += 1
                continue
            
            print(f"Processing ({index+1}/{len(df)}): {filename} ...")
            
            # Extract
            text = self.extract_text_from_url(url)
            
            # Quality Check
            if len(text) < 100:
                 print(f"  [WARNING] Low text count ({len(text)} chars). Check URL manually.")
            
            # Save
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            
            success_count += 1
            
            # Rate Limit
            time.sleep(random.uniform(0.3, 0.6))
            
        print("-" * 50)
        print(f"Extraction Complete. {success_count} files available in '{self.output_folder}'.")

if __name__ == "__main__":
    extractor = FOMCTextExtractor()
    extractor.run()