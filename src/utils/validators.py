import numpy as np

class DataValidator:
    @staticmethod
    def validate_eeg_data(data):
        """Validate EEG data quality"""
        checks = {
            'missing_values': np.isnan(data).sum() == 0,
            'signal_range': np.abs(data).max() < 500,  # typical EEG range
            'length': len(data) > 1000  # minimum required length
        }
        return all(checks.values()), checks
    
    @staticmethod
    def validate_labels(labels):
        """Validate sleep stage labels"""
        valid_stages = set(range(5))  # 0-4 for sleep stages
        return all(label in valid_stages for label in labels) 