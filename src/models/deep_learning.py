"""
Deep Learning Training Module.

Provides simplified configuration and execution wrappers surrounding TensorFlow/Keras.
Defines complex structural sequence ingestion for Recurrent Neural Networks (LSTM) 
as well as deeper nonlinear representations mapping via Dense Feed Forward layers (FNN).
Handles Class Imbalance intrinsically via configured positive class weights.
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Explicitly reduce TF C++ logging spam

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization

class DeepLearningTrainer:
    """Class orchestrating initialization, compilation, and evaluation of DL networks."""
    
    def __init__(self, model_type='fnn', input_dim=10, lr=0.001, epochs=5, batch_size=256, pos_weight=1.0):
        """
        Instantiates specific architecture configurations statically defining depth parameters.
        
        Args:
            model_type (str): 'fnn' (Feed Forward) or 'lstm' (Long Short-Term Memory).
            input_dim (int/tuple): Size bounding of expected feature shape.
            lr (float): Adam optimization learning step bound.
            epochs (int): Number of cyclic dataset passes.
            batch_size (int): Optimization memory chunk limit.
            pos_weight (float): Inverse density multiplier artificially inflating gradient responses for minority data types.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_type = model_type
        self.pos_weight = pos_weight
        
        # Sequentially map internal models explicitly
        self.model = self._build_model(model_type, input_dim)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # Logarithmic Binary BinaryCrossentropy handles independent sample probability 
        # evaluation efficiently within Keras
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    def _build_model(self, model_type, input_dim):
        """
        Dynamically constructs neural sequence layers based on specification.
        
        Args:
            model_type (str): Topology identifier ('fnn' or 'lstm').
            input_dim (int): Count of input features per row.
            
        Returns:
            tf.keras.Model: Uncompiled Keras Sequential tensor structure.
        """
        model = Sequential()
        
        if model_type == 'fnn':
            # FNN uses simplistic wide linear arrays
            model.add(tf.keras.Input(shape=(input_dim,)))
            model.add(Dense(64, activation='relu'))
            # BatchNormaliztion handles internal covariant shifts naturally
            model.add(BatchNormalization())
            # dropout randomly severs nodes specifically to fight deterministic overfitting
            model.add(Dropout(0.3)) 
            
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'lstm':
            # LSTM explicitly accounts for rolling temporal variables to calculate patterns linearly
            # Keras 3 Input layer mapping dynamically structures sequence bounds
            model.add(tf.keras.Input(shape=(None, input_dim)))
            model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
            
        else:
            raise ValueError("model_type must be strictly defined as 'fnn' or 'lstm'")
            
        return model

    def train(self, X_train, y_train):
        """
        Fits Neural layer mapping weights.
        
        Args:
            X_train (np.ndarray): Fully processed tabular or structural components.
            y_train (np.ndarray): Target vectors mapping values.
        """
        print(f"Training {self.model_type.upper()} model using TensorFlow/Keras...")
        
        # class_weight handles imbalance securely mathematically within gradient adjustments
        class_weight = {0: 1.0, 1: self.pos_weight}
        
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            verbose=1
        )

    def predict_proba(self, X):
        """
        Extracts sigmoidal 0->1 decimal responses bounded natively across the compiled network.
        
        Args:
            X (np.ndarray): Inference inputs (either 2D tabular or 3D temporal sequences).
            
        Returns:
            np.ndarray: Unified flat 1D array computing threat likelihood ratios.
        """
        preds = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        return preds.flatten()

    def save_model(self, path):
        """
        Saves internal protobuf structural representations to .keras zips.
        
        Args:
            path (str): Destination file path (e.g., 'model.keras').
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Dynamically resurrects saved topologies from native states.
        
        Args:
            path (str): Initialized file path storing model states structurally.
            
        Raises:
            FileNotFoundError: If the model file is absent from the disk structure.
        """
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError(f"Model initialization failure file {path} heavily missing.")
