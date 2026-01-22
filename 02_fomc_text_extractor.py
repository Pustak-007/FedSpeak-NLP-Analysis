"""
Project: Quantifying the Narrative
Module: 02_fomc_text_inspector_v3.py
Description: 
    DEBUG MODE. 
    Uses "Text Density" heuristic to solve the 2000-2005 HTML Table issue.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
import sys

class FOMCTextInspector:
    def __init__(self):
        self.base_dir = "data/raw_data"
        self.input_csv = os.path.join(self.base_dir, "fomc_links.csv")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def clean_text(self, text: str) -> str:
        # Collapse multiple spaces/newlines into one
        return re.sub(r'\s+', ' ', text).strip()

    def extract_text_from_url(self, url: str) -> str:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return f"[HTTP {response.status_code}]"
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # --- STEP 1: REMOVE JUNK ---
            # Kill scripts, styles, inputs, and links that are obviously navigation
            for script in soup(["script", "style", "nav", "footer", "header", "input", "button"]):
                script.decompose()

            # --- STRATEGY A: The "Golden" Containers (Modern & Mid-Era) ---
            # If these exist, they are 100% correct. Use them.
            golden_ids = ['article', 'leftText', 'content']
            for gid in golden_ids:
                div = soup.find('div', {'id': gid})
                if div:
                    # Found a known container. Extract text from it.
                    return self.clean_text(div.get_text())
            
            # --- STRATEGY B: The "Density" Heuristic (2000-2005 Fix) ---
            # If no container found, the text is hidden in a <td> or generic <div>.
            # We will find the element with the MOST text characters.
            
            candidates = []
            
            # Look at all Table Cells and Divs
            for element in soup.find_all(['td', 'div']):
                # Get raw text
                text = element.get_text()
                # Simple cleaning to get true length
                clean = " ".join(text.split())
                
                # Filter out likely navigation bars (short text, or mostly links)
                if len(clean) < 200: 
                    continue
                    
                # Store it
                candidates.append(clean)
            
            # If we found candidates, pick the longest one
            if candidates:
                # Sort by length, descending
                candidates.sort(key=len, reverse=True)
                longest_text = candidates[0]
                
                # Check if it looks like a statement
                if "Federal Open Market Committee" in longest_text or "FOMC" in longest_text or "rates" in longest_text:
                    return self.clean_text(longest_text)
                
                # If the longest text is suspicious, maybe try the second longest?
                # For now, return the longest.
                return self.clean_text(longest_text)

            # --- STRATEGY C: Desperation (Raw Text) ---
            # If tables failed, just grab everything.
            return self.clean_text(soup.get_text())

        except Exception as e:
            return f"[ERROR: {str(e)}]"

    def run(self):
        if not os.path.exists(self.input_csv):
            print(f"Error: Could not find {self.input_csv}")
            return

        df = pd.read_csv(self.input_csv)
        print("="*70)
        print(f"INSPECTING {len(df)} LINKS (Density Heuristic)")
        print("="*70)

        for index, row in df.iterrows():
            date = row['Date']
            url = row['URL']
            
            print(f"[{index+1:03d}/{len(df)}] {date} | ", end="")
            
            try:
                text = self.extract_text_from_url(url)
                
                if "ERROR" in text or "HTTP" in text:
                    print(f"FAIL -> {text}")
                elif len(text) < 200:
                    print(f"WARNING -> Short Text ({len(text)} chars)")
                else:
                    # Success print
                    print(f"OK ({len(text)} chars) | {text[:60]}...")
            
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print(f"CRITICAL FAIL -> {e}")

if __name__ == "__main__":
    inspector = FOMCTextInspector()
    inspector.run()