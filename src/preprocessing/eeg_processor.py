import numpy as np
from scipy import signal

class EEGProcessor:
    def __init__(self, config):
        self.config = config
        self.sampling_rate = config['sampling_rate']
        
    def process(self, raw_data):
        """Process raw EEG data"""
        # Get filter parameters
        low_freq = self.config['filter']['low_freq']
        high_freq = self.config['filter']['high_freq']
        
        # Get data
        data = raw_data.get_data()
        
        # Process each channel
        processed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            # Apply bandpass filter
            filtered = self._apply_bandpass_filter(data[i], low_freq, high_freq)
            
            # Normalize
            processed_data[i] = self._normalize_data(filtered)
            
        return processed_data
        
    def _apply_bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter to the data"""
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, data)
        return filtered
        
    def _normalize_data(self, data):
        """Normalize data to zero mean and unit variance"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)