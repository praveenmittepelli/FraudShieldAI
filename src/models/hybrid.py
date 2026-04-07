"""
Hybrid Ensemble module.

Orchestrates the binding of disparate model architectures (Supervised vs Deep Learning,
Standard vs Sequential) via a weighted soft-voting structure to optimize the final 
probability of fraudulent behavior.
"""

import numpy as np

class HybridEnsemble:
    def __init__(self, weights=None):
        """
        Initializes the Hybrid Ensemble.
        :param weights: A dictionary mapping model names to their weights in the soft voting ensemble.
        """
        # Default weighting: prioritize more robust models.
        self.weights = weights or {
            'LogisticRegression': 0.1,
            'RandomForest': 0.2,
            'XGBoost': 0.3,
            'FNN': 0.2,
            'LSTM': 0.2
        }

    def predict_proba(self, all_probas, sequence_indices=None):
        """
        Combines probabilities from multiple models.
        
        :param all_probas: Dictionary mapping model name to an array of predicted probabilities.
        :param sequence_indices: If sequence models (e.g., LSTM) produce fewer predictions 
                                 due to windowing, this array specifies which rows in the 
                                 standard tabular data they correspond to. 
                                 If None, assumes all arrays are the same length.
        """
        # Determine the number of standard samples based on LogisticRegression (or any standard model)
        standard_model_name = next(iter(all_probas.keys())) # Grab first model length
        n_samples = len(all_probas[standard_model_name])
        
        combined_scores = np.zeros(n_samples)
        total_weight = 0.0
        
        for name, probas in all_probas.items():
            weight = self.weights.get(name, 0)
            if weight == 0:
                continue
                
            # Handle sequence models which might have fewer predictions
            if len(probas) < n_samples and sequence_indices is not None:
                aligned_probas = np.zeros(n_samples)
                aligned_probas[sequence_indices] = probas
                # Alternatively, pad the beginning if simply sliced without indices map
                
                # For simplicity in this implementation, if it's the exact diff of seq_length
                # we just pad the beginning with the mean or 0, or just align indices
            elif len(probas) < n_samples:
                # Assuming simple sliding window truncation at the beginning
                diff = n_samples - len(probas)
                aligned_probas = np.pad(probas, (diff, 0), 'constant', constant_values=np.mean(probas))
                probas = aligned_probas
                
            combined_scores += probas * weight
            total_weight += weight
            
        return combined_scores / total_weight

    def predict(self, all_probas, threshold=0.5, sequence_indices=None):
        """Returns binary predictions based on the chosen threshold."""
        combined_scores = self.predict_proba(all_probas, sequence_indices)
        return (combined_scores >= threshold).astype(int)
