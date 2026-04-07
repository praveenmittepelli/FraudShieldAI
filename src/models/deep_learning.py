import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logs

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization

class DeepLearningTrainer:
    def __init__(self, model_type='fnn', input_dim=10, lr=0.001, epochs=5, batch_size=256, pos_weight=1.0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_type = model_type
        self.pos_weight = pos_weight
        
        self.model = self._build_model(model_type, input_dim)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # We can implement sample weighting during fit, so we use standard BinaryCrossentropy
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    def _build_model(self, model_type, input_dim):
        model = Sequential()
        
        if model_type == 'fnn':
            model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(1, activation='sigmoid'))
            
        elif model_type == 'lstm':
            # input_dim for LSTM is expected to be a tuple (seq_len, num_features)
            # but our signature passed scalar initially. Let me adapt to assume input_dim here is number of features,
            # and input_shape is (None, input_dim) to allow flexible sequence lengths
            model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, input_shape=(None, input_dim)))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
            
        else:
            raise ValueError("model_type must be 'fnn' or 'lstm'")
            
        return model

    def train(self, X_train, y_train):
        print(f"Training {self.model_type.upper()} model using TensorFlow/Keras...")
        
        # In Keras, we can pass class weights to handle imbalance directly
        class_weight = {0: 1.0, 1: self.pos_weight}
        
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            verbose=1
        )

    def predict_proba(self, X):
        # Keras predict returns probabilities (batch, 1) directly for sigmoid output
        preds = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        return preds.flatten()

    def save_model(self, path):
        # Keras models can be saved natively to .keras or directory
        self.model.save(path)

    def load_model(self, path):
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError(f"Model file {path} not found.")
