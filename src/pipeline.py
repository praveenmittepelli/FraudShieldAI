"""
Pipeline Orchestration Module for FraudShieldAI.

This module acts as the primary execution engine. It is responsible for loading the 
dataset, applying comprehensive feature engineering, slicing sequences, and instantiating
both the Deterministic Tree Models and the Sequence/Dense Neural Networks.
Following training, it automatically triggers robust evaluations and logs all reports.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engineering import FeatureEngineer
from models.supervised import SupervisedModels
from models.deep_learning import DeepLearningTrainer
from evaluation import evaluate_predictions, evaluate_models_comparison
import os

def create_sequences(df, target_col='is_fraud', seq_len=5):
    """
    Creates temporal transaction sequences sequentially indexed by User ID.
    
    LSTM Models expect data in structures representing time. This logic takes tabular 
    financial data and converts it into a rolling window sequence tensor (samples, time_steps, features).
    
    Args:
        df (pd.DataFrame): The transformed chronologically sorted dataframe.
        target_col (str): Truth label determining classification (default 'is_fraud').
        seq_len (int): Rolling lag factor defining time-steps per memory chunk.
    
    Returns:
        tuple: (np.ndarray of shape (X, 5, Features), np.ndarray of (X, 1) target labels)
    """
    sequences = []
    targets = []
    features = [c for c in df.columns if c not in [target_col, 'cc_num']]
    
    # Isolate sequences rigidly by unqiue customer to prevent cross-customer data leakage
    for cc_num, group in df.groupby('cc_num'):
        user_data = group[features].values
        user_target = group[target_col].values
        
        if len(user_data) < seq_len:
            # If the user has 3 transactions and the model needs 5, pad the remaining with pure zeroes.
            pad_size = seq_len - len(user_data)
            pad_data = np.zeros((pad_size, user_data.shape[1]))
            user_data = np.vstack([pad_data, user_data])
            targets.append(user_target[-1])
            sequences.append(user_data)
        else:
            # Extract sliding overlapping windows using a simple iterator
            for i in range(len(user_data) - seq_len + 1):
                window = user_data[i:i+seq_len]
                target = user_target[i+seq_len-1]
                sequences.append(window)
                targets.append(target)
                
    return np.array(sequences), np.array(targets)


def main():
    """
    The orchestrator execution block.
    
    Runs automatically when pipeline.py is invoked via terminal.
    Executes preprocessors, trains models, scores inferences, and exports visuals.
    """
    print("Loading data...")
    # Utilize relative dynamic pathing to ensure platform-independent execution
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
                import sys
                sys.exit(1)
            except Exception as e:
                print(f"Failed to download from Google Drive: {e}")
                import sys
                sys.exit(1)
        else:
            print("No URL provided. The pipeline may fail if the data file is missing.")
            
    df = pd.read_csv(data_path)
    
    fe = FeatureEngineer(target_col='is_fraud')
    
    print("Engineering features...")
    df_processed = fe.fit_transform(df, training=True)
    
    print("\nPreparing for Supervised...")
    # CRITICAL: We enforce a train_test_split on the `cc_num` column, NOT random rows!
    # If we split securely by Users, we ensure the LSTM and velocity metrics don't
    # accidentally bleed temporal knowledge about a user's future into the training set.
    unique_users = df_processed['cc_num'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    
    train_df = df_processed[df_processed['cc_num'].isin(train_users)].copy()
    test_df = df_processed[df_processed['cc_num'].isin(test_users)].copy()
    
    X_train = train_df.drop(columns=['is_fraud', 'cc_num'])
    y_train = train_df['is_fraud'].values
    X_test = test_df.drop(columns=['is_fraud', 'cc_num'])
    y_test = test_df['is_fraud'].values
    
    # --- 1. SUPERVISED MODELS ---
    print("\n================== 1. SUPERVISED MODELS ==================")
    os.makedirs('saved_models', exist_ok=True)
    
    sup_models = SupervisedModels()
    sup_models.train_all(X_train, y_train)
    sup_models.save_models(path_prefix='saved_models/sup_')
    
    print("\nEvaluating Supervised Models...")
    sup_preds, sup_probas = sup_models.predict_all(X_test)
    results_dict = {}
    for name in sup_preds.keys():
        evaluate_predictions(y_test, sup_preds[name], sup_probas[name], model_name=name, save_dir='saved_models/')
        results_dict[name] = {'y_pred': sup_preds[name], 'y_prob': sup_probas[name]}
        
    # --- 2. DEEP LEARNING MODELS ---
    print("\n================== 2. DEEP LEARNING MODELS ==================")
    
    # Train robust fully connected Feed Forward layers
    print("\nTraining FNN...")
    fnn = DeepLearningTrainer(model_type='fnn', input_dim=X_train.shape[1], epochs=3, batch_size=256, pos_weight=10.0)
    fnn.train(X_train.values, y_train)
    fnn.save_model('saved_models/dl_fnn.keras')
    
    print("\nEvaluating FNN...")
    fnn_prob = fnn.predict_proba(X_test.values)
    fnn_pred = (fnn_prob >= 0.5).astype(int)
    evaluate_predictions(y_test, fnn_pred, fnn_prob, model_name="FNN", save_dir='saved_models/')
    results_dict['FNN'] = {'y_pred': fnn_pred, 'y_prob': fnn_prob}
    
    # Align Temporal Sequence Array Structure 
    print("\nPreparing Sequences for LSTM...")
    X_train_seq, y_train_seq = create_sequences(train_df, seq_len=5)
    
    print("Training LSTM...")
    lstm = DeepLearningTrainer(model_type='lstm', input_dim=X_train_seq.shape[2], epochs=3, batch_size=256, pos_weight=10.0)
    lstm.train(X_train_seq, y_train_seq)
    lstm.save_model('saved_models/dl_lstm.keras')
    
    print("\nEvaluating LSTM...")
    # Calculate separate independent structures
    X_test_seq, y_test_seq = create_sequences(test_df, seq_len=5)
    lstm_prob = lstm.predict_proba(X_test_seq)
    lstm_pred = (lstm_prob >= 0.5).astype(int)
    evaluate_predictions(y_test_seq, lstm_pred, lstm_prob, model_name="LSTM", save_dir='saved_models/')
    
    # --- 3. FINAL COMPARISON ---
    print("\n================== 3. FINAL COMPARISON ==================")
    os.makedirs('plots', exist_ok=True)
    # Generates a comparison PR curve for all models sharing the same y_test matrix natively
    # print_metrics=False forcibly overrides duplicate terminal prints from populating
    evaluate_models_comparison(results_dict, y_test, save_dir='plots/', print_metrics=False)
    
    print("\nPipeline Complete. Models/reports saved to 'saved_models/' and plots saved to 'plots/'.")

if __name__ == '__main__':
    main()
