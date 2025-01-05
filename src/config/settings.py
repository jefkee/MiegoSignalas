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
    'window_size': 1000,  # 10 seconds at 100Hz
    'channels': REQUIRED_CHANNELS,
    'n_classes': 5,  # W, N1, N2, N3, REM
    'filters': [32, 64],  # Increased filters for better feature detection
    'kernel_size': 5,  # Increased kernel for better temporal patterns
    'pool_size': 2,  # Smaller pooling to preserve temporal information
    'lstm_units': 32,  # Increased units for better sequence learning
    'dropout_rate': 0.3,
    'learning_rate': 0.0005  # Reduced for better stability
}

# Training parameters  
TRAINING_PARAMS = {
    'batch_size': 32,  # Smaller batches for better generalization
    'epochs': 10,  # Increased epochs for better learning
    'validation_split': 0.2
}

# Preprocessing parameters
PREPROCESSING_PARAMS = {
    'window_size': 10,  # seconds - matches model window_size
    'sampling_rate': 100,  # Hz
    'overlap': 0.5,  # Added overlap for better temporal continuity
    'lowpass': 30,  # Hz
    'highpass': 0.3  # Hz
} 