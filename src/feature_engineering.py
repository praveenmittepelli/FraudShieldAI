
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in miles between two points 
    on the earth (specified in decimal degrees)
    """
    R = 3958.8 # Earth radius in miles
    
    # Convert latitude and longitude to radians
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

class FeatureEngineer:
    def __init__(self, target_col='is_fraud', sample_frac=0.1):
        self.target_col = target_col
        self.sample_frac = sample_frac
        self.scaler = StandardScaler()
        self.target_encoder = ce.TargetEncoder()
        
    def calculate_rolling_features(self, df):
        print("Calculating rolling features per cc_num...")
        # Sort by cc_num and time
        df = df.sort_values(by=['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
        
        # Amount rolled
        df['amt_rolling_mean_7d'] = df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df['amt_rolling_std_7d'] = df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0)
        
        # Velocity (transaction count past 7 days isn't trivial with rolling index without time index,
        # but we can use simple shift or rolling window size if we assume transactions are proxy for time, 
        # or just rolling count of last N trades).
        df['txn_velocity_7'] = df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window=7, min_periods=1).count())
        
        return df

    def fit_transform(self, df, training=True):
        print(f"Initial shape: {df.shape}")
        
        # Downsample majority class aggressively if training
        if training:
            frauds = df[df[self.target_col] == 1]
            non_frauds = df[df[self.target_col] == 0]
            
            print(f"Frauds: {len(frauds)}, Non-Frauds: {len(non_frauds)}")
            
            # To preserve sequence logic, if we downsample now we destroy the cc_num sequences.
            # To handle this, we sample specific users or we just use the last year of data.
            # Let's say we randomly select 20,000 non-fraud users instead of sampling rows.
            # Actually, to make it fast and simple, we'll sample the dataframe
            # but keep sequences intact for selected cards. Let's sample cc_nums instead of rows.
            unique_non_fraud_cc = non_frauds['cc_num'].unique()
            sample_size = min(2000, len(unique_non_fraud_cc))
            sampled_non_fraud_cc = np.random.choice(unique_non_fraud_cc, size=sample_size, replace=False)
            fraud_cc = frauds['cc_num'].unique()
            selected_ccs = np.concatenate([fraud_cc, sampled_non_fraud_cc])
            
            df = df[df['cc_num'].isin(selected_ccs)].copy()
            print(f"Shape after sampling users: {df.shape}")
        
        # Convert timestamp
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        # Calculate Rolling
        df = self.calculate_rolling_features(df)
        
        # Spatial features
        print("Calculating geospatial features...")
        df['distance_miles'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
        
        # Temporal features
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        
        # Age
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
        
        # Drop raw categorical and dates we won't encode quickly
        cols_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time']
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Encode
        categorical_cols = ['merchant', 'category', 'gender', 'job']
        
        if training:
            print("Encoding categories and scaling...")
            df[categorical_cols] = self.target_encoder.fit_transform(df[categorical_cols], df[self.target_col])
            
            # Scale
            features = df.drop(columns=[self.target_col, 'cc_num'])
            scaled_features = self.scaler.fit_transform(features)
            df[features.columns] = scaled_features
            
            # Save artifacts
            dump(self.target_encoder, 'target_encoder.pkl')
            dump(self.scaler, 'scaler.pkl')
            
        else:
            self.target_encoder = load('target_encoder.pkl')
            self.scaler = load('scaler.pkl')
            
            # Replace unknown target encodings with global mean (default behavior of category_encoders)
            df[categorical_cols] = self.target_encoder.transform(df[categorical_cols])
            
            features = df.drop(columns=[self.target_col, 'cc_num'], errors='ignore')
            scaled_features = self.scaler.transform(features)
            df[features.columns] = scaled_features
            
        return df
