"""
Project: Quantifying the Narrative
Module: 01_fomc_link_scraper_v8_sorted.py
Description: 
    Scrapes FOMC Statement links (2000-2024).
    NEW: Extracts exact dates from URLs and sorts the final CSV chronologically.
    OUTPUT: 'data/raw_data/fomc_links.csv' (Sorted by Date)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
from typing import List, Dict, Optional

class FOMCLinkScraper:
    def __init__(self):
        self.base_url = "https://www.federalreserve.gov"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.output_dir = "data/raw_data"
        self.output_filename = "fomc_links.csv"

    def _get_soup_with_fallback(self, year: int) -> Optional[BeautifulSoup]:
        urls_to_try = [
            f"https://www.federalreserve.gov/monetarypolicy/fomccalendars{year}.htm",
            f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
        ]
        if year >= 2019:
            urls_to_try.append("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")

        for url in urls_to_try:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return BeautifulSoup(response.content, 'html.parser')
            except requests.exceptions.RequestException:
                continue
        return None

    def _extract_year_from_url(self, url: str) -> Optional[int]:
        match = re.search(r'(20\d{2})', url)
        if match:
            return int(match.group(1))
        return None

    def _extract_full_date(self, url: str) -> str:
        """
        Extracts YYYYMMDD and converts to YYYY-MM-DD for sorting.
        """
        # Look for 8 digits: 20000202
        match = re.search(r'(20\d{2})(\d{2})(\d{2})', url)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return "Unknown"

    def get_links_for_year(self, target_year: int) -> List[Dict]:
        print(f"Scanning Year: {target_year}...")
        soup = self._get_soup_with_fallback(target_year)
        if not soup:
            return []

        links_found = []
        for anchor in soup.find_all('a'):
            href = anchor.get('href')
            text = anchor.get_text().strip()
            if not href: continue
            
            href_lower = href.lower()
            text_lower = text.lower()
            
            # --- FILTER LOGIC ---
            is_statement = "statement" in text_lower or \
                           ("monetary" in href_lower and "pressreleases" in href_lower and "a.htm" in href_lower)

            if not is_statement: continue

            exclusion_keywords = ["minutes", "press conference", "call", "video", "pdf", 
                                  "financial", "longer-run", "implementation", "correction", 
                                  "projection", "balance sheet"]
            if any(k in text_lower for k in exclusion_keywords): continue

            # Year Validation
            url_year = self._extract_year_from_url(href)
            if url_year and url_year != target_year: continue
            if target_year >= 2015 and not url_year: continue

            # Path Validation
            valid_paths = ["/newsevents/press/monetary", "/boarddocs/press/general", 
                           "/boarddocs/press/monetary", "/newsevents/pressreleases/monetary"]
            if not any(p in href_lower for p in valid_paths): continue

            full_url = self.base_url + href if href.startswith('/') else href
            
            # Extract Sortable Date
            sortable_date = self._extract_full_date(full_url)

            links_found.append({
                "Date": sortable_date,
                "Year": target_year,
                "Date_Description": text, 
                "URL": full_url
            })
        
        # Local Deduplication
        unique_links = []
        seen = set()
        for link in links_found:
            if link['URL'] not in seen:
                unique_links.append(link)
                seen.add(link['URL'])
        return unique_links

    def save_data(self, df: pd.DataFrame):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        full_path = os.path.join(self.output_dir, self.output_filename)
        df.to_csv(full_path, index=False)
        print(f"\n[SUCCESS] Saved Sorted Manifest to: {full_path}")

    def run(self, start_year: int, end_year: int):
        all_data = []
        for year in range(start_year, end_year + 1):
            year_data = self.get_links_for_year(year)
            all_data.extend(year_data)
            
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            # Drop duplicates
            df = df.drop_duplicates(subset=['URL'])
            
            # --- THE SORTING FIX ---
            # Sort by the new 'Date' column
            df = df.sort_values(by='Date').reset_index(drop=True)
            
            print(f"\nTotal Links Found: {len(df)}")
            print("\n--- FIRST 5 ROWS (Check Dates) ---")
            print(df[['Date', 'URL']].head())
            print("\n--- LAST 5 ROWS (Check Dates) ---")
            print(df[['Date', 'URL']].tail())
            
            self.save_data(df)
        else:
            print("Scraper failed.")

if __name__ == "__main__":
    scraper = FOMCLinkScraper()
    scraper.run(start_year=2000, end_year=2024)