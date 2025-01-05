import tensorflow as tf
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, 
    LSTM, Dense, BatchNormalization, GlobalAveragePooling1D
)
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam

class SleepClassifier:
    """Neural network classifier for sleep stage scoring"""
    
    def __init__(self, model_params):
        """Initialize the classifier with model parameters"""
        self.model = self._build_model(model_params)
        
    def _build_model(self, params):
        """Build the neural network model
        
        The model outputs probabilities for 5 sleep stages in order:
        0 = Wake
        1 = N1 (Light sleep)
        2 = N2 (Intermediate sleep)
        3 = N3 (Deep sleep)
        4 = REM
        """
        model = Sequential([
            Conv1D(filters=params['filters'][0], 
                  kernel_size=params['kernel_size'],
                  activation='relu',
                  input_shape=(params['window_size'], len(params['channels']))),
            MaxPooling1D(pool_size=params['pool_size']),
            
            Conv1D(filters=params['filters'][1],
                  kernel_size=params['kernel_size'],
                  activation='relu'),
            MaxPooling1D(pool_size=params['pool_size']),
            
            Conv1D(filters=params['filters'][2],
                  kernel_size=params['kernel_size'], 
                  activation='relu'),
            MaxPooling1D(pool_size=params['pool_size']),
            
            LSTM(params['lstm_units'], return_sequences=True),
            Dropout(params['dropout_rate']),
            
            LSTM(params['lstm_units']),
            Dropout(params['dropout_rate']),
            
            Dense(5, activation='softmax')  # 5 classes: Wake, N1, N2, N3, REM
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, eeg_data):
        """Predict sleep stages from EEG data
        
        Returns probabilities for each stage in order:
        [Wake, N1, N2, N3, REM]
        """
        predictions = self.model.predict(eeg_data)
        return predictions