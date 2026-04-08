"""
Feature Engineering module for FraudShieldAI.

This module encompasses the data transformation pipeline. It is responsible for:
1. Calculating geospatial distances (Haversine formula).
2. Generating rolling velocity metrics to detect rapid account takeover behavior.
3. Encoding categorical variables using Target Encoding.
4. Downsampling logic to balance training data while preserving user sequence integrity.
"""

import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in miles between two points 
    on the earth (specified in decimal degrees).
    
    This is highly indicative of card-not-present (CNP) fraud or 
    physical theft if a transaction occurs far from the user's home state.
    
    Args:
        lat1 (pd.Series): Latitude of the customer's home.
        lon1 (pd.Series): Longitude of the customer's home.
        lat2 (pd.Series): Latitude of the merchant.
        lon2 (pd.Series): Longitude of the merchant.
        
    Returns:
        pd.Series: The calculated distance in miles.
    """
    R = 3958.8 # Earth radius in miles
    
    # Convert latitude and longitude to radians for trigonometric mapping
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Apply Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

class FeatureEngineer:
    """
    An orchestrator class that handles stateful transformations (Scaling and Encoding)
    and applies stateless transformations (Temporal extraction, Geospatial math).
    """
    def __init__(self, target_col='is_fraud', sample_frac=0.1):
        """
        Initializes the transformers required for the pipeline.
        
        Args:
            target_col (str): The classification column name.
            sample_frac (float): The ratio to downsample if operating dynamically.
        """
        self.target_col = target_col
        self.sample_frac = sample_frac
        self.scaler = StandardScaler()
        self.target_encoder = ce.TargetEncoder()
        
    def calculate_rolling_features(self, df):
        """
        Computes sequential rolling timeframe features on a per-user basis.
        
        Args:
            df (pd.DataFrame): The raw transaction dataframe.
            
        Returns:
            pd.DataFrame: The dataframe augmented with rolling velocity/mean metrics.
        """
        print("Calculating rolling features per cc_num...")
        # Sort chronologically per user to ensure rolling windows are accurate
        df = df.sort_values(by=['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
        
        # Determine average amount spent in the recent 7 transactions
        df['amt_rolling_mean_7d'] = df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        # Determine volatility (standard deviation) of recent spending
        df['amt_rolling_std_7d'] = df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0)
        
        # Calculate velocity proxy (count of transactions in recent window)
        df['txn_velocity_7'] = df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).count())
        
        return df

    def fit_transform(self, df, training=True):
        """
        Executes the full preprocessing pipeline on the input dataframe.
        
        Args:
            df (pd.DataFrame): Raw transaction data.
            training (bool): If True, fits the Scaler and Encoder, and aggressively downsamples
                             the majority class. If False, only transforms data.
                             
        Returns:
            pd.DataFrame: Fully processed, scaled, and encoded numerical dataframe ready for inference.
        """
        print(f"Initial shape: {df.shape}")
        
        # Conditionally downsample majority class (Non-Frauds) if training
        if training:
            frauds = df[df[self.target_col] == 1]
            non_frauds = df[df[self.target_col] == 0]
            
            print(f"Frauds: {len(frauds)}, Non-Frauds: {len(non_frauds)}")
            
            # To preserve LSTM logic, we sample entire credit card numbers (users) 
            # instead of individual rows. This prevents breaking the timeline sequence.
            unique_non_fraud_cc = non_frauds['cc_num'].unique()
            sample_size = min(2000, len(unique_non_fraud_cc))
            sampled_non_fraud_cc = np.random.choice(unique_non_fraud_cc, size=sample_size, replace=False)
            fraud_cc = frauds['cc_num'].unique()
            
            selected_ccs = np.concatenate([fraud_cc, sampled_non_fraud_cc])
            df = df[df['cc_num'].isin(selected_ccs)].copy()
            print(f"Shape after sampling users: {df.shape}")
        
        # Convert raw string timestamp to datetimes
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        # Apply Temporal Rolling Mathematics
        df = self.calculate_rolling_features(df)
        
        # Calculate Physical Deviation (Geospatial)
        print("Calculating geospatial features...")
        df['distance_miles'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
        
        # Feature Extract Temporal properties
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        
        # Feature Extract User Age
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
        
        # Prune dataset of PII, raw strings, and non-predictive identifiers
        cols_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time']
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Prepare categorical high-cardinality features 
        categorical_cols = ['merchant', 'category', 'gender', 'job']
        
        if training:
            print("Encoding categories and scaling...")
            # Use TargetEncoding to convert high cardinality strings to robust floats mapped against the target
            df[categorical_cols] = self.target_encoder.fit_transform(df[categorical_cols], df[self.target_col])
            
            # Use StandardScaler to bring magnitude of large metrics (amt, distance) in line with standard models (LogReg, FNN)
            features = df.drop(columns=[self.target_col, 'cc_num'])
            scaled_features = self.scaler.fit_transform(features)
            df[features.columns] = scaled_features
            
            # Persist Transformers to disk for deployment
            dump(self.target_encoder, 'saved_models/target_encoder.pkl')
            dump(self.scaler, 'saved_models/scaler.pkl')
            
        else:
            # Load stored Transformers in Inference mode
            # We explicitly load these dynamically in a real scenario
            try:
                self.target_encoder = load('saved_models/target_encoder.pkl')
                self.scaler = load('saved_models/scaler.pkl')
            except FileNotFoundError:
                print("Warning: Missing scaler/encoder at 'saved_models/' in inference mode.")
            
            # Apply transforming logic
            df[categorical_cols] = self.target_encoder.transform(df[categorical_cols])
            
            features = df.drop(columns=[self.target_col, 'cc_num'], errors='ignore')
            scaled_features = self.scaler.transform(features)
            df[features.columns] = scaled_features
            
        return df
