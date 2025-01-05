import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
    @property
    def model_params(self):
        return self.config['model']
        
    @property
    def preprocessing_params(self):
        return self.config['preprocessing'] 

REQUIRED_CHANNELS = [
    'EEG Fpz-Cz',  # Main EEG channel
    'EEG Pz-Oz',   # Secondary EEG channel
    'EOG horizontal',  # Eye movement
    'EMG submental'    # Muscle activity
]

# Model parameters
MODEL_PARAMS = {
    'input_shape': (1500, len(REQUIRED_CHANNELS)),  # 15 seconds at 100Hz (reduced from 30s)
    'n_classes': 5,  # W, N1, N2, N3, REM
    'filters': [16, 32, 64],  # Reduced filter sizes
    'kernel_size': 5,  # Reduced kernel size
    'pool_size': 4,  # Increased pool size for faster reduction
    'lstm_units': 64,  # Reduced LSTM units
    'dropout_rate': 0.2
}

# Training parameters  
TRAINING_PARAMS = {
    'batch_size': 128,  # Increased batch size for faster processing
    'epochs': 50,  # Reduced epochs
    'validation_split': 0.2,
    'learning_rate': 0.001  # Slightly increased for faster convergence
}

# Preprocessing parameters
PREPROCESSING_PARAMS = {
    'window_size': 15,  # seconds (reduced from 30s)
    'sampling_rate': 100,  # Hz
    'overlap': 0.25,  # Reduced overlap for fewer segments
    'lowpass': 30,  # Hz
    'highpass': 0.3  # Hz
} 