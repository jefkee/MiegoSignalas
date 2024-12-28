import numpy as np
from scipy import signal
import scipy.stats

class FeatureExtractor:
    """Extract features from EEG signals"""
    
    def extract_features(self, eeg_data):
        """Extract features from MNE Raw object"""
        # Get data as numpy array and convert to microvolts
        data = eeg_data.get_data() * 1e6  # Convert to microvolts
        
        # Add debug print
        print(f"\nSignal ranges before processing:")
        for ch_idx, ch_name in enumerate(eeg_data.ch_names):
            ch_data = data[ch_idx, :]
            print(f"{ch_name}: min={ch_data.min():.2f}, max={ch_data.max():.2f}, "
                  f"mean={ch_data.mean():.2f}, std={ch_data.std():.2f}")
        
        features = {}
        for ch_idx, ch_name in enumerate(eeg_data.ch_names):
            # Get single channel data
            ch_data = data[ch_idx, :]
            
            # Skip channels with no variation
            if ch_data.std() < 1e-6:
                print(f"Warning: Channel {ch_name} has very low variation")
                continue
            
            features[ch_name] = {
                'raw_data': ch_data,  # Add raw data
                'power_bands': self._calculate_power_bands(ch_data),
                'statistical': self._calculate_statistical_features(ch_data)
            }
        
        return features
    
    def _calculate_power_bands(self, data, fs=100):
        """Calculate EEG frequency bands for a single channel"""
        # Scale the data to microvolts
        data = data * 1e6
        
        bands = {
            'delta': (0.5, 4),    # Delta band
            'theta': (4, 8),      # Theta band
            'alpha': (8, 13),     # Alpha band
            'beta': (13, 30)      # Beta band
        }
        
        powers = {}
        for band_name, (low, high) in bands.items():
            # Apply bandpass filter
            filtered = self._bandpass_filter(data, low, high, fs)
            # Calculate power using Welch's method
            freqs, psd = signal.welch(filtered, fs, nperseg=fs*2)
            # Get the mean power in the band
            idx = np.logical_and(freqs >= low, freqs <= high)
            powers[band_name] = np.mean(psd[idx])
        
        return powers
    
    def _calculate_statistical_features(self, data):
        """Calculate statistical features for a single channel"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'kurtosis': scipy.stats.kurtosis(data),
            'skewness': scipy.stats.skew(data)
        }
    
    def _bandpass_filter(self, data, low, high, fs):
        """Apply bandpass filter to single channel data"""
        nyquist = fs / 2
        b, a = signal.butter(4, [low/nyquist, high/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, data)
        return filtered