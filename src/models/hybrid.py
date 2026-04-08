"""
Hybrid Ensemble Model Module.

This architectural component orchestrates the binding of disparate model structures 
(Classical Scikit-Learn algorithms, XGBoost, and Deep Learning Sequences) 
via a weighted soft-voting matrix to optimize the final classification of extreme fraud events.
"""

import numpy as np

class HybridEnsemble:
    """
    Manages predictions from individual distinct architectures and merges them together, 
    accounting for mismatched sequence shapes and confidence offsets.
    """
    def __init__(self, weights=None):
        """
        Initializes the Hybrid Ensemble orchestration container.
        
        Args:
            weights (dict): A dictionary explicitly mapping model names to their importance 
                            weights in the final soft voting ensemble score. 
                            If none provided, defaults exist.
        """
        # Default weighting matrix: 
        # Deep Learning comprises 40% (FNN 20%, LSTM 20%) while 
        # Decision Trees comprises 50% (XGBoost 30%, RF 20%).
        self.weights = weights or {
            'LogisticRegression': 0.1,
            'RandomForest': 0.2,
            'XGBoost': 0.3,
            'FNN': 0.2,
            'LSTM': 0.2
        }

    def predict_proba(self, all_probas, sequence_indices=None):
        """
        Combines probabilities functionally across multi-variate shape matrices.
        
        Args:
            all_probas (dict): Directory mapping model labels to numpy arrays of predicted fraud scores.
            sequence_indices (np.ndarray): LSTM models use rolling sequence arrays. If seq_length=5, 
                                           the first 4 targets are dropped. Sequences_indices specifies 
                                           which robust row corresponds to which tabular element.
                                           
        Returns:
            np.ndarray: Unified risk score matrix matched sequentially per row.
        """
        # Establish base structural constraints driven from a 'standard' tabular output shape
        standard_model_name = next(iter(all_probas.keys())) 
        n_samples = len(all_probas[standard_model_name])
        
        combined_scores = np.zeros(n_samples)
        total_weight = 0.0
        
        for name, probas in all_probas.items():
            weight = self.weights.get(name, 0)
            if weight == 0:
                # Do not calculate null-weighted trees
                continue
                
            # If sequence arrays are mismatched in size from standard arrays
            if len(probas) < n_samples and sequence_indices is not None:
                aligned_probas = np.zeros(n_samples)
                aligned_probas[sequence_indices] = probas
                probas = aligned_probas
                
            # Naive fallback: Simple truncation padding calculation assuming the 
            # delta loss exists directly at the array origin (start)
            elif len(probas) < n_samples:
                diff = n_samples - len(probas)
                # Pad the missing introductory transactions with the model mean to prevent wild variance shifts
                aligned_probas = np.pad(probas, (diff, 0), 'constant', constant_values=np.mean(probas))
                probas = aligned_probas
                
            # Accumulate the weighted product sum
            combined_scores += probas * weight
            total_weight += weight
            
        # Return probability distributed between 0 and 1
        return combined_scores / total_weight

    def predict(self, all_probas, threshold=0.5, sequence_indices=None):
        """
        Maps probability bounds explicitly to binary deterministic classifications.
        
        Args:
            all_probas (dict): Base dictionaries holding arrays of probabilities.
            threshold (float): Decision bound limit. Default standard is 50%.
            sequence_indices (np.ndarray): Pointers for LSTM mapping structure.
            
        Returns:
            np.ndarray: Binary array denoting 0 (Genunine) or 1 (Fraudulent).
        """
        combined_scores = self.predict_proba(all_probas, sequence_indices)
        return (combined_scores >= threshold).astype(int)
