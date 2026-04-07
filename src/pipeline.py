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
    Creates temporal sequences for LSTM by cc_num.
    Assumes df is already sorted chronologically per cc_num.
    """
    sequences = []
    targets = []
    features = [c for c in df.columns if c not in [target_col, 'cc_num']]
    
    for cc_num, group in df.groupby('cc_num'):
        user_data = group[features].values
        user_target = group[target_col].values
        
        if len(user_data) < seq_len:
            pad_size = seq_len - len(user_data)
            pad_data = np.zeros((pad_size, user_data.shape[1]))
            user_data = np.vstack([pad_data, user_data])
            targets.append(user_target[-1])
            sequences.append(user_data)
        else:
            for i in range(len(user_data) - seq_len + 1):
                window = user_data[i:i+seq_len]
                target = user_target[i+seq_len-1]
                sequences.append(window)
                targets.append(target)
                
    return np.array(sequences), np.array(targets)

def main():
    print("Loading data...")
    # Dynamic absolute pathing against CWD
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'fraudTrain.csv')
    df = pd.read_csv(data_path)
    
    fe = FeatureEngineer(target_col='is_fraud')
    
    print("Engineering features...")
    df_processed = fe.fit_transform(df, training=True)
    
    print("\nPreparing for Supervised...")
    # Train test split on users to avoid data leak
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
    
    print("\nTraining FNN...")
    fnn = DeepLearningTrainer(model_type='fnn', input_dim=X_train.shape[1], epochs=3, batch_size=256, pos_weight=10.0)
    fnn.train(X_train.values, y_train)
    fnn.save_model('saved_models/dl_fnn.keras')
    
    print("\nEvaluating FNN...")
    fnn_prob = fnn.predict_proba(X_test.values)
    fnn_pred = (fnn_prob >= 0.5).astype(int)
    evaluate_predictions(y_test, fnn_pred, fnn_prob, model_name="FNN", save_dir='saved_models/')
    results_dict['FNN'] = {'y_pred': fnn_pred, 'y_prob': fnn_prob}
    
    print("\nPreparing Sequences for LSTM...")
    X_train_seq, y_train_seq = create_sequences(train_df, seq_len=5)
    
    print("Training LSTM...")
    lstm = DeepLearningTrainer(model_type='lstm', input_dim=X_train_seq.shape[2], epochs=3, batch_size=256, pos_weight=10.0)
    lstm.train(X_train_seq, y_train_seq)
    lstm.save_model('saved_models/dl_lstm.keras')
    
    print("\nEvaluating LSTM...")
    X_test_seq, y_test_seq = create_sequences(test_df, seq_len=5)
    lstm_prob = lstm.predict_proba(X_test_seq)
    lstm_pred = (lstm_prob >= 0.5).astype(int)
    evaluate_predictions(y_test_seq, lstm_pred, lstm_prob, model_name="LSTM", save_dir='saved_models/')
    
    # --- 3. FINAL COMPARISON ---
    print("\n================== 3. FINAL COMPARISON ==================")
    os.makedirs('plots', exist_ok=True)
    # Generates a comparison PR curve for all models sharing the same y_test
    # print_metrics=False prevents re-evaluating and re-logging metrics/reports
    evaluate_models_comparison(results_dict, y_test, save_dir='plots/', print_metrics=False)
    
    print("\nPipeline Complete. Models/reports saved to 'saved_models/' and plots saved to 'plots/'.")

if __name__ == '__main__':
    main()
