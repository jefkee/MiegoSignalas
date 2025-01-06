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
        
        # Load all PSG files with progress bar
        for psg_file in tqdm(self._get_psg_files(), desc="Loading EDF files"):
            try:
                # Load EDF
                raw_data = mne.io.read_raw_edf(psg_file, preload=True)
                
                # Get corresponding hypnogram
                hypno_file = self._get_hypnogram_file(psg_file)
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
                            segment = segment.T  # Shape: (3000, n_channels)
                            
                            # Preprocess
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
                print(f"Error processing {psg_file}: {e}")
                continue
        
        if not features:
            raise ValueError("No valid data loaded")
        
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
        base_name = os.path.basename(psg_file).replace('-PSG.edf', '')
        
        # Try different possible hypnogram names
        patterns = [
            '-Hypnogram.edf',
            '.Hypnogram.edf',
            '-HYP.edf',
            '.HYP.edf',
            '_HYP.edf',
            '-hyp.edf',
            '.hyp.edf',
            '_hyp.edf'
        ]
        
        # Try with both original name and J/0 variants
        base_variants = [
            base_name,
            base_name.replace('0', 'J'),
            base_name.replace('J', '0')
        ]
        
        for base in base_variants:
            for pattern in patterns:
                hypno_name = base + pattern
                full_path = os.path.join(self.data_dir, hypno_name)
                if os.path.exists(full_path):
                    return full_path
        
        # If no matching file found, return the default name
        return os.path.join(self.data_dir, base_name + '-Hypnogram.edf')
    
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