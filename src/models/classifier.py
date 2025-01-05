import tensorflow as tf
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Dense, 
    LSTM, BatchNormalization, Bidirectional,
    GlobalAveragePooling1D, Attention
)
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

class SleepClassifier:
    """Neural network classifier for sleep stage scoring"""
    
    def __init__(self, model_params):
        """Initialize the classifier with model parameters"""
        self.model = self._build_model(model_params)
        # Adjusted class weights based on typical sleep stage distributions
        self.class_weights = {
            0: 2.0,    # Wake: increase weight to prevent REM dominance
            1: 4.0,    # N1: rare stage, needs higher weight
            2: 1.0,    # N2: most common stage
            3: 2.0,    # N3: increase weight for deep sleep
            4: 2.0     # REM: balanced weight
        }
        
    def _build_model(self, params):
        """Build the neural network model with enhanced architecture"""
        model = Sequential([
            # Input layer
            Input(shape=(params['window_size'], len(params['channels']))),
            
            # First convolutional block - temporal feature extraction
            Conv1D(filters=params['filters'][0], 
                  kernel_size=params['kernel_size'],
                  activation='relu',
                  padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Second convolutional block - frequency feature extraction
            Conv1D(filters=params['filters'][1],
                  kernel_size=params['kernel_size'],
                  activation='relu',
                  padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Bidirectional LSTM for sequence learning
            Bidirectional(LSTM(params['lstm_units'], 
                             return_sequences=True,
                             dropout=0.2,
                             recurrent_dropout=0.2)),
            BatchNormalization(),
            
            # Global average pooling to reduce sequence length
            GlobalAveragePooling1D(),
            
            # Dense classification layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer with balanced initialization
            Dense(params['n_classes'], 
                  activation='softmax',
                  bias_initializer='zeros',
                  kernel_initializer='glorot_uniform')
        ])
        
        # Use a lower learning rate for better stability
        optimizer = Adam(learning_rate=params['learning_rate'])
        
        # Compile with class weights
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, **kwargs):
        """Train the model with class weights"""
        return self.model.fit(
            X_train, y_train,
            class_weight=self.class_weights,
            **kwargs
        )
    
    def predict(self, eeg_data):
        """Predict sleep stages from EEG data"""
        # Add temperature scaling for better calibrated predictions
        predictions = self.model.predict(eeg_data)
        # Apply softmax temperature scaling (T=2.0 for softer predictions)
        predictions = np.exp(np.log(predictions) / 2.0)
        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
        return predictions