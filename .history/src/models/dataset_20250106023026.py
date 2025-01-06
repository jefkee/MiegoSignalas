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
        
        # Define required channels and expected shapes
        required_channels = [
            'EEG Fpz-Cz',
            'EEG Pz-Oz',
            'EOG horizontal',
            'EMG submental'
        ]
        
        SEGMENT_SIZE = 3000  # 30 seconds @ 100Hz
        N_CHANNELS = len(required_channels)
        EXPECTED_SHAPE = (SEGMENT_SIZE, N_CHANNELS)
        
        # Get list of PSG files
        psg_files = self._get_psg_files()
        print(f"\nFound {len(psg_files)} PSG files in {self.data_dir}")
        
        if not psg_files:
            raise ValueError(f"No PSG files found in directory: {self.data_dir}")
        
        # Load all PSG files with progress bar
        for psg_file in tqdm(psg_files, desc="Loading EDF files"):
            try:
                print(f"\nProcessing {os.path.basename(psg_file)}")
                
                # Load EDF
                raw_data = mne.io.read_raw_edf(psg_file, preload=True)
                print(f"Loaded EDF with {len(raw_data.ch_names)} channels")
                print(f"Available channels: {raw_data.ch_names}")
                
                # Check required channels
                missing_channels = [ch for ch in required_channels if ch not in raw_data.ch_names]
                if missing_channels:
                    print(f"Warning: Missing required channels in {os.path.basename(psg_file)}: {missing_channels}")
                    continue
                    
                # Select only required channels in specific order
                raw_data.pick_channels(required_channels)
                
                # Get corresponding hypnogram
                hypno_file = self._get_hypnogram_file(psg_file)
                if hypno_file is None:
                    print(f"Warning: No matching hypnogram found for {os.path.basename(psg_file)}")
                    continue
                    
                if not os.path.exists(hypno_file):
                    print(f"Warning: Hypnogram file not found: {hypno_file}")
                    continue
                    
                sleep_stages = self._load_sleep_stages(hypno_file)
                print(f"Loaded {len(sleep_stages)} sleep stages")
                
                if len(sleep_stages) == 0:
                    print(f"Warning: No valid sleep stages found in {hypno_file}")
                    continue
                
                # Get raw data
                data = raw_data.get_data()  # Shape: (n_channels, n_times)
                print(f"Raw data shape: {data.shape}")
                
                # Verify data dimensions
                if data.shape[0] != N_CHANNELS:
                    print(f"Warning: Unexpected number of channels: {data.shape[0]}, expected {N_CHANNELS}")
                    continue
                
                # Create segments
                n_segments = min(len(sleep_stages), data.shape[1] // SEGMENT_SIZE)
                print(f"Creating {n_segments} segments")
                
                file_features = []
                file_labels = []
                
                for i in range(n_segments):
                    try:
                        start_idx = i * SEGMENT_SIZE
                        end_idx = start_idx + SEGMENT_SIZE
                        
                        if end_idx <= data.shape[1]:
                            # Extract and transpose segment
                            segment = data[:, start_idx:end_idx].T  # Shape: (3000, n_channels)
                            
                            # Verify segment shape
                            if segment.shape != EXPECTED_SHAPE:
                                print(f"Warning: Invalid segment shape: {segment.shape}, expected {EXPECTED_SHAPE}")
                                continue
                                
                            # Preprocess
                            try:
                                segment = self._preprocess_segment(segment)
                                
                                # Verify preprocessed shape
                                if segment.shape != EXPECTED_SHAPE:
                                    print(f"Warning: Invalid shape after preprocessing: {segment.shape}")
                                    continue
                    start_idx = i * segment_size
                    end_idx = start_idx + segment_size
                    
                    if end_idx <= data.shape[1]:  # Only if we have enough data
                        segment = data[:, start_idx:end_idx]
                        segment = segment.T  # Shape: (3000, n_channels)
                        
                        # Verify segment shape
                        if segment.shape != (3000, len(required_channels)):
                            print(f"Warning: Invalid segment shape: {segment.shape}")
                            continue
                        
                        try:
                            # Preprocess
                            segment = self._preprocess_segment(segment)
                            
                            # Add original
                            features.append(segment)
                            labels.append(sleep_stages[i])
                            
                            # Add augmented version if enabled
                            if self.use_augmentation:
                                aug_segment = self._augment_segment(segment.copy())
                                features.append(aug_segment)
                                labels.append(sleep_stages[i])
                        except Exception as e:
                            print(f"Error processing segment {i}: {str(e)}")
                            continue
            
            except Exception as e:
                print(f"Error processing {psg_file}: {str(e)}")
                continue
        
        if not features:
            raise ValueError(f"No valid data loaded from {len(psg_files)} PSG files. Check the data directory and file formats.")
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Print overall class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nOverall class distribution:")
        for stage, count in zip(unique, counts):
            print(f"Stage {stage}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        print(f"\nFinal dataset shapes:")
        print(f"Features: {features.shape}")
        print(f"Labels: {labels.shape}")
        
        return features, labels
    
    def _get_psg_files(self):
        """Get list of PSG files"""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        psg_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                     if f.endswith('-PSG.edf')]
        
        if not psg_files:
            print(f"\nNo PSG files found in {self.data_dir}")
            print("Files in directory:", os.listdir(self.data_dir))
        
        return psg_files
    
    def _get_hypnogram_file(self, psg_file):
        """Get corresponding hypnogram file for a PSG file"""
        base_name = os.path.basename(psg_file).replace('-PSG.edf', '')
        
        # Print debug info
        print(f"\nLooking for hypnogram file for {base_name}")
        print(f"Directory contents: {os.listdir(self.data_dir)}")
        
        # Map of known PSG to Hypnogram file patterns
        known_mappings = {
            'SC4001E0': 'SC4001EC',
            'SC4002E0': 'SC4002EC',
            'SC4011E0': 'SC4011EH',
            'SC4012E0': 'SC4012EC'
        }
        
        # First try known mapping
        if base_name in known_mappings:
            hypno_name = f"{known_mappings[base_name]}-Hypnogram.edf"
            full_path = os.path.join(self.data_dir, hypno_name)
            if os.path.exists(full_path):
                print(f"Found matching hypnogram: {hypno_name}")
                return full_path
        
        # Try different possible hypnogram names
        patterns = [
            '-Hypnogram.edf',
            'C-Hypnogram.edf',  # For EC pattern
            'H-Hypnogram.edf',  # For EH pattern
            '.Hypnogram.edf',
            '-HYP.edf',
            '.HYP.edf',
            '_HYP.edf',
            '-hyp.edf',
            '.hyp.edf',
            '_hyp.edf'
        ]
        
        # Try with both original name and variants
        base_variants = [
            base_name,
            base_name[:-1] + 'C',  # Replace last char with C
            base_name[:-1] + 'H',  # Replace last char with H
            base_name.replace('0', 'C'),  # Replace 0 with C
            base_name.replace('0', 'H')   # Replace 0 with H
        ]
        
        for base in base_variants:
            for pattern in patterns:
                hypno_name = base + pattern
                full_path = os.path.join(self.data_dir, hypno_name)
                if os.path.exists(full_path):
                    print(f"Found matching hypnogram: {hypno_name}")
                    return full_path
                else:
                    print(f"Tried but not found: {hypno_name}")
        
        # If no matching file found, return None instead of default name
        print(f"No matching hypnogram found for {base_name}")
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