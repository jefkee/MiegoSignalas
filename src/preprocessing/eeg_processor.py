import mne
import numpy as np
from typing import Dict, List, Tuple

class EEGProcessor:
    """
    Handles EEG signal processing for sleep data
    """
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        
    def preprocess_signal(self, raw_eeg: mne.io.Raw) -> mne.io.Raw:
        """
        Apply basic preprocessing to EEG signal
        """
        # Bandpass filter (0.3-35 Hz for sleep EEG)
        raw_eeg.filter(l_freq=0.3, h_freq=35)
        
        return raw_eeg