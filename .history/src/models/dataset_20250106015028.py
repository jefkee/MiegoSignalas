import os
import mne
import numpy as np
from tqdm import tqdm
from typing import Tuple
import scipy.signal as signal

class SleepDataset:
    def __init__(self, data_dir, use_augmentation=False):
        self.data_dir = data_dir
        self.use_augmentation = use_augmentation
        
    def load_training_data(self):
        """Load and prepare training data"""
        features = []
        labels = []
        
        # Required EEG channels for sleep staging
        required_channels = [
            'EEG Fpz-Cz',   # Frontal EEG
            'EEG Pz-Oz',    # Occipital EEG
            'EOG horizontal',# Horizontal EOG
            'EMG submental', # Chin EMG
            'EMG chin1',     # Alternative chin EMG name
            'EMG chin2',     # Alternative chin EMG name
            'ECG'           # ECG signal
        ]
        
        # Load all PSG files with progress bar
        for psg_file in tqdm(self._get_psg_files(), desc="Loading EDF files"):
            try:
                # Load EDF
                raw_data = mne.io.read_raw_edf(psg_file, preload=True)
                print(f"\nAvailable channels in {os.path.basename(psg_file)}:")
                print(raw_data.ch_names)
                
                # Select required channels that are available
                available_channels = []
                for ch in required_channels:
                    if ch in raw_data.ch_names:
                        available_channels.append(ch)
                
                if len(available_channels) < 4:  # Minimum required channels
                    print(f"Skipping {psg_file} - insufficient channels")
                    continue
                
                # Pick only the available required channels
                raw_data.pick_channels(available_channels)
                print(f"Selected channels: {raw_data.ch_names}")
                
                # Get corresponding hypnogram
                hypno_file = self._get_hypnogram_file(psg_file)
                if hypno_file is None:
                    print(f"Skipping {psg_file} - no matching hypnogram found")
                    continue
                    
                sleep_stages = self._load_sleep_stages(hypno_file)
                
                if len(sleep_stages) > 0:
                    # Print class distribution for this file
                    unique, counts = np.unique(sleep_stages, return_counts=True)
                    print(f"\nClass distribution for {os.path.basename(psg_file)}:")
                    for stage, count in zip(unique, counts):
                        print(f"Stage {stage}: {count} samples")
                    
                    # Get raw data and reshape
                    data = raw_data.get_data()  # Shape: (n_channels, n_times)
                    
                    # Create segments of 30 seconds (3000 samples at 100Hz)
                    segment_size = 3000
                    n_segments = len(sleep_stages)
                    
                    for i in range(n_segments):
                        start_idx = i * segment_size
                        end_idx = start_idx + segment_size
                        
                        if end_idx <= data.shape[1]:  # Only if we have enough data
                            segment = data[:, start_idx:end_idx]
                            
                            # Ensure consistent channel count by padding if necessary
                            if segment.shape[0] < len(required_channels):
                                pad_channels = len(required_channels) - segment.shape[0]
                                segment = np.pad(segment, ((0, pad_channels), (0, 0)), 
                                               mode='constant', constant_values=0)
                            
                            segment = segment.T  # Shape: (3000, n_channels)
                            
                            # Preprocess
                            try:
                                segment = self._preprocess_segment(segment)
                                
                                # Add original
                                features.append(segment)
                                labels.append(sleep_stages[i])
                                
                                # Add augmented version
                                if self.use_augmentation:
                                    aug_segment = self._augment_segment(segment.copy())
                                    features.append(aug_segment)
                                    labels.append(sleep_stages[i])
                            except Exception as e:
                                print(f"Error preprocessing segment {i}: {e}")
                                continue
            
            except Exception as e:
                print(f"Error processing {psg_file}: {e}")
                continue
        
        if not features:
            raise ValueError("No valid data loaded. Check that PSG files have matching hypnograms.")
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Print overall class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nOverall class distribution:")
        for stage, count in zip(unique, counts):
            print(f"Stage {stage}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        print(f"\nFeatures shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return features, labels
    
    def _get_psg_files(self):
        """Get list of PSG files"""
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                if f.endswith('-PSG.edf')]
    
    def _get_hypnogram_file(self, psg_file):
        """Get corresponding hypnogram file for a PSG file"""
        # Get base name by removing '-PSG.edf'
        base_name = os.path.basename(psg_file).replace('-PSG.edf', '')
        print(f"\nLooking for hypnogram for PSG file: {base_name}")
        
        # Replace E0 with EC or EH
        possible_bases = [
            base_name.replace('E0', 'EC'),
            base_name.replace('E0', 'EH')
        ]
        
        # Try each possible base name
        for base in possible_bases:
            hypno_name = f"{base}-Hypnogram.edf"
            full_path = os.path.join(self.data_dir, hypno_name)
            if os.path.exists(full_path):
                print(f"Found matching hypnogram: {hypno_name}")
                return full_path
        
        print(f"Warning: No matching hypnogram found for {base_name}")
        print("Available files in directory:", os.listdir(self.data_dir))
        return None
    
    def _load_sleep_stages(self, hypno_file):
        """Load sleep stages from hypnogram file"""
        try:
            if not os.path.exists(hypno_file):
                print(f"Warning: Hypnogram file not found: {hypno_file}")
                return np.array([])
            
            annot = mne.read_annotations(hypno_file)
            stages = []
            for description in annot.description:
                if description in ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 
                                 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R']:
                    stage_map = {
                        'Sleep stage W': 0,
                        'Sleep stage 1': 1,
                        'Sleep stage 2': 2,
                        'Sleep stage 3': 3,
                        'Sleep stage 4': 3,  # Combine stages 3 and 4 as N3
                        'Sleep stage R': 4
                    }
                    stages.append(stage_map[description])
            return np.array(stages)
        except Exception as e:
            print(f"Error loading hypnogram {hypno_file}: {e}")
            return np.array([]) 
    
    def _preprocess_segment(self, segment):
        """Preprocess a single segment"""
        try:
            # Normalize each channel independently
            for i in range(segment.shape[1]):
                channel = segment[:, i]
                # Remove mean and scale to unit variance
                channel = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
                segment[:, i] = channel
            
            # Apply bandpass filter (0.5-40 Hz)
            fs = 100  # sampling frequency
            segment = mne.filter.filter_data(
                segment.T, fs, l_freq=0.5, h_freq=40, method='iir'
            ).T
            
            return segment
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def _augment_segment(self, segment):
        """Apply random augmentations"""
        # Gaussian noise
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.0001, 0.0005)
            segment += np.random.normal(0, noise_level, segment.shape)
        
        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            segment *= scale
        
        # Time warping
        if np.random.random() < 0.3:
            time_warp = np.random.uniform(0.9, 1.1)
            new_len = int(segment.shape[0] * time_warp)
            segment = signal.resample(segment, new_len)
            if new_len > segment.shape[0]:
                segment = segment[:segment.shape[0]]
            else:
                pad_len = segment.shape[0] - new_len
                segment = np.pad(segment, ((0, pad_len), (0, 0)))
        
        return segment