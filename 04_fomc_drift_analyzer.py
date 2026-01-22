"""
Project: Quantifying the Narrative
Module: 04_fomc_drift_analyzer.py
Description: 
    Calculates 'Semantic Drift' (Cosine Distance) between consecutive statements.
    High Drift = Significant change in language (Potential Market Shock).
    OUTPUT: 'data/modified_data/fomc_drift.csv'
"""

import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FOMCDriftEngine:
    def __init__(self):
        self.raw_dir = "data/raw_data/statements"
        self.output_dir = "data/modified_data"
        self.output_file = os.path.join(self.output_dir, "fomc_drift.csv")

    def load_statements(self):
        """Loads all text files into a DataFrame sorted by date."""
        files = [f for f in os.listdir(self.raw_dir) if f.endswith(".txt")]
        files.sort() # Critical: Ensure chronological order
        
        data = []
        for filename in files:
            date_str = filename.replace(".txt", "")
            filepath = os.path.join(self.raw_dir, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Clean numbers to focus on linguistic changes
            # (e.g., changing "2 percent" to "3 percent" is a small vector change, 
            # but changing "robust" to "weak" is a big one)
            text_clean = re.sub(r'\d+', '', text)
            
            data.append({"Date": date_str, "Text": text_clean})
            
        return pd.DataFrame(data)

    def calculate_drift(self, df):
        """
        Computes Cosine Similarity between T and T-1.
        Drift = 1 - Similarity.
        """
        # TF-IDF Vectorizer
        # max_df=0.95: Ignore words that appear in 95% of documents (e.g., "Federal", "Reserve")
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        
        tfidf_matrix = vectorizer.fit_transform(df['Text'])
        
        drift_scores = [0.0] # First meeting has no history
        
        print(f"Calculating Drift for {len(df)} statements...")
        
        for i in range(1, len(df)):
            # Compare Today (i) vs Yesterday (i-1)
            vec_t = tfidf_matrix[i]
            vec_prev = tfidf_matrix[i-1]
            
            similarity = cosine_similarity(vec_t, vec_prev)[0][0]
            drift = 1 - similarity
            drift_scores.append(drift)
            
        df['Drift_Score'] = drift_scores
        return df

    def run(self):
        df = self.load_statements()
        print(f"Loaded {len(df)} statements.")
        
        df_drift = self.calculate_drift(df)
        # Save only the metrics, not the massive text blob
        output_df = df_drift[['Date', 'Drift_Score']]
        output_df.to_csv(self.output_file, index=False)
        print("-" * 50)
        print(f"Drift Analysis Complete. Saved to: {self.output_file}")
        print("First 5 Rows:")
        print(output_df.head())

if __name__ == "__main__":
    engine = FOMCDriftEngine()
    engine.run()