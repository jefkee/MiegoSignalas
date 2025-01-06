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
    'EEG Fpz-Cz',
    'EEG Pz-Oz',
    'EOG horizontal',
    'EMG submental'
]

# Model parameters
MODEL_PARAMS = {
    'input_shape': (3000, len(REQUIRED_CHANNELS)),  # 30 seconds at 100Hz
    'n_classes': 5,
    'filters': [16, 32, 64],
    'kernel_size': 3,
    'pool_size': 2,
    'lstm_units': 64,
    'dropout_rate': 0.3
}

# Training parameters  
TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'learning_rate': 0.001
}

# Preprocessing parameters
PREPROCESSING_PARAMS = {
    'window_size': 30,  # seconds
    'sampling_rate': 100,  # Hz
    'overlap': 0.5  # 50% overlap between windows
} 