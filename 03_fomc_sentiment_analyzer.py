"""
Project: Quantifying the Narrative
Module: 03_fomc_sentiment_analyzer.py
Description: 
    Loads extracted text from 'data/raw_data/statements/'.
    Initializes the FinBERT model (ProsusAI).
    Scores every sentence for Positive/Negative/Neutral sentiment.
    Aggregates scores into a 'Fed Sentiment Index'.
    OUTPUT: 'data/modified_data/fomc_sentiment.csv'
"""

import pandas as pd
import os
import torch
import nltk
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tqdm import tqdm

# --- FIX FOR NLTK ERROR ---
# We must download the sentence splitter rules
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK 'punkt_tab' data...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' data...")
    nltk.download('punkt')

class FOMCSentimentEngine:
    def __init__(self):
        # --- PATHS ---
        self.raw_dir = "data/raw_data/statements"
        self.output_dir = "data/modified_data"
        self.output_file = os.path.join(self.output_dir, "fomc_sentiment.csv")
        
        # --- AI CONFIGURATION ---
        print("Initializing FinBERT Model...")
        self.model_name = "ProsusAI/finbert"
        
        # Determine Device
        if torch.backends.mps.is_available():
            self.device = 0 # MPS (Mac)
            print("Using Hardware Acceleration (MPS/Mac)")
        elif torch.cuda.is_available():
            self.device = 0 # CUDA
            print("Using Hardware Acceleration (CUDA)")
        else:
            self.device = -1 # CPU
            print("Using CPU")

        # Load the Pipeline
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=self.device)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess_text(self, text):
        """
        Splits text into sentences and removes administrative junk.
        """
        # This sent_tokenize is what requires punkt_tab
        sentences = nltk.sent_tokenize(text)
        clean_sentences = []
        
        for sent in sentences:
            if len(sent) < 20: continue
            
            admin_phrases = [
                "Voting for the FOMC", "Voting against", "release date", 
                "For immediate release", "Board of Governors", "monetary policy action"
            ]
            if any(phrase in sent for phrase in admin_phrases):
                continue
                
            clean_sentences.append(sent)
            
        return clean_sentences

    def score_statement(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        sentences = self.preprocess_text(text)
        
        if not sentences:
            return None

        # Run AI on all sentences
        # Truncation=True handles sentences longer than 512 tokens (rare but possible)
        results = self.nlp(sentences, truncation=True, max_length=512)
        
        pos_count = 0
        neg_count = 0
        neu_count = 0
        
        for res in results:
            label = res['label']
            if label == 'positive':
                pos_count += 1
            elif label == 'negative':
                neg_count += 1
            else:
                neu_count += 1
        
        # The "Fedspeak Index"
        total_relevant = pos_count + neg_count + neu_count
        net_sentiment = (pos_count - neg_count) / total_relevant if total_relevant > 0 else 0
        
        return {
            "num_sentences": total_relevant,
            "sentiment_score": net_sentiment,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "neu_count": neu_count
        }

    def run(self):
        files = [f for f in os.listdir(self.raw_dir) if f.endswith(".txt")]
        files.sort()
        
        print(f"Found {len(files)} statements. Starting Analysis...")
        
        data_rows = []
        
        for filename in tqdm(files):
            filepath = os.path.join(self.raw_dir, filename)
            stats = self.score_statement(filepath)
            
            if stats:
                date_str = filename.replace(".txt", "")
                
                row = {
                    "Date": date_str,
                    "Sentiment_Score": stats['sentiment_score'],
                    "Positive": stats['pos_count'],
                    "Negative": stats['neg_count'],
                    "Neutral": stats['neu_count'],
                    "Total_Sentences": stats['num_sentences']
                }
                data_rows.append(row)
                
        df = pd.DataFrame(data_rows)
        df.to_csv(self.output_file, index=False)
        print("-" * 50)
        print(f"Analysis Complete. Data saved to: {self.output_file}")
        print("First 5 Rows:")
        print(df.head())

if __name__ == "__main__":
    engine = FOMCSentimentEngine()
    engine.run()