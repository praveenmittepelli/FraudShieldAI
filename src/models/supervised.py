"""
Supervised Learning Module.

Wraps classical structural machine learning models (Logistic Regression, Random Forest, XGBoost).
These models act as fundamentally interpretable foundations against dense neural boundaries.
All configurations explicitly balance classes to address <1% Fraud ratios natively!
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import pandas as pd
import joblib

class SupervisedModels:
    """Class orchestrator supporting execution of deterministic Decision Trees and basic regressions."""
    
    def __init__(self, random_state=42):
        """
        Declares all supervised architectures and hyperparameter bounds.
        
        Args:
            random_state (int): Enforce standard deterministic state replication.
        """
        self.random_state = random_state
        self.models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, 
                random_state=random_state, 
                # Scikit-learn will automatically inversely adjust matrix weighting to data density
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=random_state, 
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, 
                max_depth=5, 
                learning_rate=0.1, 
                random_state=random_state, 
                eval_metric='logloss',
                # XGBoost specific parameter scaling: scale_pos_weight is approx the count
                # of negative ratio over positive ratio. 99 represents roughly ~1% fraud.
                scale_pos_weight=99
            )
        }
        self.trained_models = {}
        
    def train_all(self, X_train, y_train):
        """
        Iteratively trains all instantiated baseline structural models.
        
        Args:
            X_train (np.ndarray/pd.DataFrame): Normalized feature structures.
            y_train (np.ndarray): Binary Truths.
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        print("Finished training all supervised models.")

    def predict_all(self, X_test):
        """
        Requests probability metrics against inference set for all saved models.
        
        Args:
            X_test (np.ndarray/pd.DataFrame): Validation features.
            
        Returns:
            tuple: (dict mapping prediction binary arrays, dict mapping probability continuous arrays).
        """
        predictions = {}
        probas = {}
        for name, model in self.trained_models.items():
            predictions[name] = model.predict(X_test)
            
            # Trees support probabilistic distribution logic via predict_proba directly.
            # SVMs and older architectures usually only have decision functions.
            if hasattr(model, "predict_proba"):
                probas[name] = model.predict_proba(X_test)[:, 1]
            else:
                probas[name] = model.decision_function(X_test)
        return predictions, probas

    def save_models(self, path_prefix='saved_models/sup_'):
        """
        Saves all trained supervised models to disk efficiently utilizing joblib threading.
        
        Args:
            path_prefix (str): Storage path identifying string locations for export.
        """
        for name, model in self.trained_models.items():
            filename = f"{path_prefix}{name.lower()}.joblib"
            print(f"Saving {name} to {filename}...")
            joblib.dump(model, filename)

def get_supervised_predictions(X_train, y_train, X_test):
    """
    Stand-alone utility function to immediately execute train/test procedures asynchronously.
    """
    sm = SupervisedModels()
    sm.train_all(X_train, y_train)
    predictions, probas = sm.predict_all(X_test)
    return sm, predictions, probas
