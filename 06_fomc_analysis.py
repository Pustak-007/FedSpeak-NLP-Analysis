"""
Project: Quantifying the Narrative
Module: 06_fomc_analysis.py
Description: 
    Loads the Master Dataset.
    Runs OLS Regression (Sentiment vs. Market).
    Generates High-Quality Visualizations saved to 'figures/'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

class FOMCAnalysis:
    def __init__(self):
        # --- PATHS ---
        # Input comes from data/modified_data
        self.base_dir = "data/modified_data"
        self.input_file = os.path.join(self.base_dir, "fomc_full_dataset.csv")
        
        # Output goes to 'figures' (Sibling to 'data')
        self.figures_dir = "figures"
        
        # Create Figures Directory
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
            print(f"[+] Created directory: {self.figures_dir}")

        # Plotting Style (Professional)
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'serif' 

    def load_data(self):
        if not os.path.exists(self.input_file):
            print("Error: Dataset not found. Run Module 05.")
            return None
        
        df = pd.read_csv(self.input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def run_regression(self, df):
        """
        Runs OLS Regression: 
        Y = Market Reaction (2Y Yield Change)
        X = Sentiment Score + Drift
        """
        print("\n" + "="*50)
        print("REGRESSION ANALYSIS: DOES SENTIMENT MOVE YIELDS?")
        print("="*50)
        
        # Prepare Data
        # We drop NaNs (e.g., if market data is missing for a specific day)
        df_reg = df.dropna(subset=['US2Y_Change', 'Sentiment_Score'])
        
        X = df_reg[['Sentiment_Score', 'Drift_Score']]
        X = sm.add_constant(X) # Adds Beta_0 (Intercept)
        Y = df_reg['US2Y_Change']
        
        model = sm.OLS(Y, X).fit()
        print(model.summary())
        
        # Save Summary to Text File
        with open(os.path.join(self.figures_dir, "regression_results.txt"), "w") as f:
            f.write(model.summary().as_text())

    def plot_sentiment_timeline(self, df):
        """
        Chart 1: The 'Fed Sentiment Index' (Time Series)
        """
        plt.figure(figsize=(12, 6))
        
        # Plot Line
        sns.lineplot(data=df, x='Date', y='Sentiment_Score', color='#2c3e50', linewidth=1.5, label='Raw Sentiment')
        
        # Add Smooth Moving Average
        df['MA_4'] = df['Sentiment_Score'].rolling(window=4).mean()
        sns.lineplot(data=df, x='Date', y='MA_4', color='#e74c3c', linewidth=2, label='6-Month Trend')
        
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        plt.title('The Fed Sentiment Index (2000-2024)\nQuantifying Hawkishness (Positive) vs. Dovishness (Negative)', fontsize=14, fontweight='bold')
        plt.ylabel('Net Sentiment Score (FinBERT)', fontsize=12)
        plt.xlabel('Year', fontsize=12)
        plt.legend()
        
        # Save
        save_path = os.path.join(self.figures_dir, "01_fed_sentiment_timeline.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[+] Saved Chart 1: {save_path}")

    def plot_regression_scatter(self, df):
        """
        Chart 2: Scatter Plot (Sentiment vs. Yield Change)
        """
        plt.figure(figsize=(10, 6))
        
        sns.regplot(data=df, x='Sentiment_Score', y='US2Y_Change', 
                    scatter_kws={'alpha':0.5, 'color':'#3498db'}, 
                    line_kws={'color':'#e74c3c'})
        
        plt.title('Impact of Fed Sentiment on 2-Year Treasury Yields\n(Same-Day Reaction)', fontsize=14, fontweight='bold')
        plt.xlabel('Fed Sentiment Score (Hawkish ->)', fontsize=12)
        plt.ylabel('2-Year Yield Change (Basis Points)', fontsize=12)
        
        save_path = os.path.join(self.figures_dir, "02_sentiment_yield_regression.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[+] Saved Chart 2: {save_path}")

    def plot_drift_volatility(self, df):
        """
        Chart 3: Drift vs. Absolute Volatility
        """
        df['Abs_Yield_Change'] = df['US2Y_Change'].abs()
        
        # Bin Data into Low vs High Drift
        median_drift = df['Drift_Score'].median()
        df['Drift_Regime'] = np.where(df['Drift_Score'] > median_drift, 'High Surprise', 'Low Surprise')
        
        plt.figure(figsize=(8, 6))
        
        # Using errorbar=None to remove confidence intervals for cleaner look
        sns.barplot(data=df, x='Drift_Regime', y='Abs_Yield_Change', palette='viridis', errorbar=None)
        
        plt.title('Does "Changing Language" Cause Volatility?\n(Avg Absolute Yield Move by Drift Regime)', fontsize=14, fontweight='bold')
        plt.ylabel('Avg Absolute 2Y Yield Change (bps)', fontsize=12)
        plt.xlabel('Statement Drift (Cosine Distance)', fontsize=12)
        
        save_path = os.path.join(self.figures_dir, "03_drift_volatility_impact.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[+] Saved Chart 3: {save_path}")

    def run(self):
        df = self.load_data()
        if df is None: return
        
        self.run_regression(df)
        self.plot_sentiment_timeline(df)
        self.plot_regression_scatter(df)
        self.plot_drift_volatility(df)
        
        print("\nAnalysis Complete. Check the 'figures' folder.")

if __name__ == "__main__":
    analyzer = FOMCAnalysis()
    analyzer.run()