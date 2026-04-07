"""
Supervised learning models module.

Wraps classical machine learning models (Logistic Regression, Random Forest, XGBoost)
configured specifically for imbalanced learning using appropriate class weights.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import pandas as pd
import joblib

class SupervisedModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, 
                random_state=random_state, 
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
                scale_pos_weight=99 # Approximation for highly imbalanced, can be tuned
            )
        }
        self.trained_models = {}
        
    def train_all(self, X_train, y_train):
        """Trains all configured supervised models."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        print("Finished training all supervised models.")

    def predict_all(self, X_test):
        """Returns predictions for all trained models."""
        predictions = {}
        probas = {}
        for name, model in self.trained_models.items():
            predictions[name] = model.predict(X_test)
            # Support predict_proba. fallback to decision_function if not available
            if hasattr(model, "predict_proba"):
                probas[name] = model.predict_proba(X_test)[:, 1]
            else:
                probas[name] = model.decision_function(X_test)
        return predictions, probas

    def save_models(self, path_prefix='saved_models/sup_'):
        """Saves all trained supervised models to disk using joblib."""
        for name, model in self.trained_models.items():
            filename = f"{path_prefix}{name.lower()}.joblib"
            print(f"Saving {name} to {filename}...")
            joblib.dump(model, filename)

def get_supervised_predictions(X_train, y_train, X_test):
    """Utility function to quickly train and get predictions."""
    sm = SupervisedModels()
    sm.train_all(X_train, y_train)
    predictions, probas = sm.predict_all(X_test)
    return sm, predictions, probas

