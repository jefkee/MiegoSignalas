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
        
        print("\n=== Debug Information ===")
        print(f"Number of required channels: {len(required_channels)}")
        
        # Load all PSG files with progress bar
        for psg_file in tqdm(self._get_psg_files(), desc="Loading EDF files"):
            try:
                # Load EDF
                raw_data = mne.io.read_raw_edf(psg_file, preload=True)
                print(f"\nProcessing file: {os.path.basename(psg_file)}")
                print(f"Sampling rate: {raw_data.info['sfreq']} Hz")
                print(f"Available channels ({len(raw_data.ch_names)}):")
                for ch in raw_data.ch_names:
                    print(f"  - {ch}")
                
                # Select required channels that are available
                available_channels = []
                for ch in required_channels:
                    if ch in raw_data.ch_names:
                        available_channels.append(ch)
                
                print(f"\nFound {len(available_channels)} matching channels:")
                for ch in available_channels:
                    print(f"  - {ch}")
                
                if len(available_channels) < 4:  # Minimum required channels
                    print(f"Skipping {psg_file} - insufficient channels")
                    continue
                
                # Pick only the available required channels
                raw_data.pick_channels(available_channels)
                data = raw_data.get_data()
                print(f"\nRaw data shape after channel selection: {data.shape}")
                
                # Get corresponding hypnogram
                hypno_file = self._get_hypnogram_file(psg_file)
                if hypno_file is None:
                    print(f"Skipping {psg_file} - no matching hypnogram found")
                    continue
                    
                sleep_stages = self._load_sleep_stages(hypno_file)
                
                if len(sleep_stages) > 0:
                    print(f"\nNumber of sleep stages: {len(sleep_stages)}")
                    
                    # Calculate samples per epoch based on sampling rate
                    samples_per_epoch = int(raw_data.info['sfreq'] * 30)  # 30 seconds
                    print(f"Samples per epoch: {samples_per_epoch}")
                    
                    for i in range(len(sleep_stages)):
                        start_idx = i * samples_per_epoch
                        end_idx = start_idx + samples_per_epoch
                        
                        if end_idx <= data.shape[1]:  # Only if we have enough data
                            segment = data[:, start_idx:end_idx]
                            
                            # Ensure consistent channel count by padding if necessary
                            if segment.shape[0] < len(required_channels):
                                pad_channels = len(required_channels) - segment.shape[0]
                                segment = np.pad(segment, ((0, pad_channels), (0, 0)), 
                                               mode='constant', constant_values=0)
                            
                            segment = segment.T  # Shape: (samples_per_epoch, n_channels)
                            
                            # Preprocess
                            try:
                                segment = self._preprocess_segment(segment)
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
        
        print("\n=== Final Shapes ===")
        print(f"Number of features: {len(features)}")
        print(f"First feature shape: {features[0].shape}")
        print(f"Last feature shape: {features[-1].shape}")
        
        # Convert to numpy arrays - use object dtype for varying lengths
        features = np.array(features, dtype=object)
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
        
        return segment