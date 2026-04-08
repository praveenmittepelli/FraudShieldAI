"""
Exploratory Data Analysis (EDA) Module.

Generates insightful visualizations to contextualize the severe class imbalance,
transaction amount distributions, temporal (time-of-day) trends, and feature 
correlations inherent to the financial tabular data. All artifacts are saved to a 
designated output directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import sys
import os

# Create seaborn style
plt_sns.set_theme(style="whitegrid", context="talk")

def main():
    """
    Main routine for the EDA pipeline.
    
    Reads the target dataset, evaluates basic missing properties, and orchestrates 
    the generation of multiple Seaborn / Matplotlib plots to visually represent 
    the underlying fraud structures.
    """
    if len(sys.argv) < 2:
        print("Usage: python src/eda.py <output_dir>")
        sys.exit(1)
        
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'fraudTrain.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}.")
        gdrive_url = input("Please enter the Google Drive sharing URL to download the dataset (or press Enter to skip): ").strip()
        if gdrive_url:
            try:
                import gdown
                print("Downloading dataset from Google Drive (this may take a moment)...")
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                gdown.download(gdrive_url, data_path, quiet=False, fuzzy=True)
            except ImportError:
                print("Error: The 'gdown' package is required to download from Google Drive.")
                print("Please install it by opening a new terminal and running: uv add gdown")
                sys.exit(1)
            except Exception as e:
                print(f"Failed to download from Google Drive: {e}")
                sys.exit(1)
        else:
            print("No URL provided. The pipeline may fail if the data file is missing.")
            
    df = pd.read_csv(data_path)
    
    print(f"Data Shape: {df.shape}")
    
    # Check basic missing values (simulating the EDA cleaning phase check)
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    
    print("Generating Class Imbalance Plot...")
    plt.figure(figsize=(8, 6))
    ax = plt_sns.countplot(data=df, x='is_fraud', palette='viridis')
    plt.title('Severe Class Imbalance: Legitimate vs Fraudulent', weight='bold')
    plt.xlabel('Is Fraud (0 = No, 1 = Yes)')
    plt.ylabel('Count (Log Scale)')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'class_imbalance.png'), dpi=150)
    plt.close()
    
    print("Generating Amount Distribution Plot...")
    plt.figure(figsize=(10, 6))
    # We clip amt to 99th percentile for visualization to avoid extreme outliers stretching the plot
    p99 = df['amt'].quantile(0.995)
    subset = df[df['amt'] < p99]
    plt_sns.boxplot(data=subset, x='is_fraud', y='amt', palette='muted')
    plt.title('Transaction Amount Distribution by Class (Clipped to 99.5th Pct)', weight='bold')
    plt.xlabel('Is Fraud')
    plt.ylabel('Transaction Amount ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'amount_distribution.png'), dpi=150)
    plt.close()
    
    print("Generating Temporal Trends Plot...")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    
    hourly_fraud = df[df['is_fraud'] == 1].groupby('trans_hour').size()
    hourly_legit = df[df['is_fraud'] == 0].groupby('trans_hour').size()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot legitimate
    color = 'tab:blue'
    ax1.set_xlabel('Hour of Day (0-23)')
    ax1.set_ylabel('Legitimate Volume', color=color)
    ax1.plot(hourly_legit.index, hourly_legit.values, color=color, linewidth=2, label='Legit')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Fraud Volume', color=color)  
    ax2.plot(hourly_fraud.index, hourly_fraud.values, color=color, linewidth=2, linestyle='--', label='Fraud')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Transaction Volumes over the Time of Day', weight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'temporal_trends.png'), dpi=150)
    plt.close()
    
    print("Generating Category Fraud Rate...")
    plt.figure(figsize=(12, 6))
    fraud_rate_cat = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False) * 100
    plt_sns.barplot(x=fraud_rate_cat.values, y=fraud_rate_cat.index, palette='rocket')
    plt.title('Fraud Probability by Merchant Category', weight='bold')
    plt.xlabel('Percentage of Transactions that are Fraud (%)')
    plt.ylabel('Merchant Category')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'category_fraud.png'), dpi=150)
    plt.close()
    
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['Unnamed: 0', 'unix_time', 'trans_num'], errors='ignore')
    corr = numeric_df.corr()
    plt_sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Numeric Features Correlation Heatmap', weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    print(f"EDA successfully completed. Artifacts saved to {out_dir}")

if __name__ == '__main__':
    main()
