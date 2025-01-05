import numpy as np
from scipy import signal

class FeatureExtractor:
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        
    def extract_features(self, eeg_data):
        """Extract basic features from EEG data"""
        features = {}
        
        # Process each channel
        for ch_idx, ch_name in enumerate(eeg_data.ch_names):
            ch_data = eeg_data.get_data()[ch_idx]
            
            # Skip channels with no variation
            if np.std(ch_data) < 1e-6:
                continue
                
            features[ch_name] = {
                'power_bands': self._calculate_power_bands(ch_data),
                'statistical': self._calculate_statistical_features(ch_data)
            }
            
        return features
        
    def _calculate_power_bands(self, data):
        """Calculate basic EEG frequency bands"""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        powers = {}
        for band_name, (low, high) in bands.items():
            # Apply bandpass filter
            filtered = self._bandpass_filter(data, low, high)
            # Calculate power
            powers[band_name] = np.mean(filtered ** 2)
            
        return powers
        
    def _calculate_statistical_features(self, data):
        """Calculate basic statistical features"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }
        
    def _bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter to data"""
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, data)
        return filtered