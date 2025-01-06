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
                print(f"\nProcessing PSG file: {psg_file}")
                # Load EDF
                raw_data = mne.io.read_raw_edf(psg_file, preload=True)
                print(f"Channels found: {raw_data.ch_names}")
                
                # Get corresponding hypnogram
                hypno_file = self._get_hypnogram_file(psg_file)
                print(f"Looking for hypnogram: {hypno_file}")
                sleep_stages = self._load_sleep_stages(hypno_file)
                print(f"Number of sleep stages loaded: {len(sleep_stages)}")
                
                if len(sleep_stages) > 0:
                    # Print class distribution for this file
                    unique, counts = np.unique(sleep_stages, return_counts=True)
                    print(f"Class distribution for {os.path.basename(psg_file)}:")
                    for stage, count in zip(unique, counts):
                        print(f"Stage {stage}: {count} samples")
                    
                    # Get raw data and reshape
                    data = raw_data.get_data()  # Shape: (n_channels, n_times)
                    print(f"Raw data shape: {data.shape}")
                    
                    # Create segments of 30 seconds (3000 samples at 100Hz)
                    segment_size = 3000
                    n_segments = len(sleep_stages)
                    
                    for i in range(n_segments):
                        start_idx = i * segment_size
                        end_idx = start_idx + segment_size
                        
                        if end_idx <= data.shape[1]:  # Only if we have enough data
                            segment = data[:, start_idx:end_idx]
                            segment = segment.T  # Shape: (3000, n_channels)
                            
                            # Verify segment shape
                            if segment.shape != (3000, 7):  # Changed to expect 7 channels
                                print(f"Skipping segment with incorrect shape: {segment.shape}")
                                continue
                            
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
                else:
                    print(f"No sleep stages found for {os.path.basename(psg_file)}")
            
            except Exception as e:
                print(f"Error processing {psg_file}: {str(e)}")
                continue
        
        if not features:
            raise ValueError("No valid data loaded")
        
        # Convert to numpy arrays
        features = np.stack(features)  # Using stack instead of array to ensure consistent shapes
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
        base_name = os.path.basename(psg_file)
        # Replace E0 with EC/EH and -PSG.edf with -Hypnogram.edf
        hypno_variants = [
            base_name.replace('E0-PSG.edf', 'EC-Hypnogram.edf'),
            base_name.replace('E0-PSG.edf', 'EH-Hypnogram.edf')
        ]
        
        for hypno_name in hypno_variants:
            full_path = os.path.join(self.data_dir, hypno_name)
            if os.path.exists(full_path):
                return full_path
        
        # If no matching file found, return the EC variant as default
        return os.path.join(self.data_dir, base_name.replace('E0-PSG.edf', 'EC-Hypnogram.edf'))
    
    def _load_sleep_stages(self, hypno_file):
        """Load sleep stages from hypnogram file"""
        try:
            if not os.path.exists(hypno_file):
                print(f"Warning: Hypnogram file not found: {hypno_file}")
                return np.array([])
            
            print(f"Loading annotations from: {hypno_file}")
            annot = mne.read_annotations(hypno_file)
            print(f"Found {len(annot.description)} annotations")
            print(f"Unique annotation descriptions: {np.unique(annot.description)}")
            
            stages = []
            stage_map = {
                'Sleep stage W': 0,
                'Sleep stage 1': 1,
                'Sleep stage 2': 2,
                'Sleep stage 3': 3,
                'Sleep stage 4': 3,  # Combine stages 3 and 4 as N3
                'Sleep stage R': 4,
                # Add alternative formats
                'W': 0,
                'N1': 1,
                'N2': 2,
                'N3': 3,
                'N4': 3,
                'REM': 4,
                'R': 4,
                # Add numeric formats
                '0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 3,
                '5': 4  # Some datasets use 5 for REM
            }
            
            for description in annot.description:
                desc_str = str(description).strip()
                if desc_str in stage_map:
                    stages.append(stage_map[desc_str])
                else:
                    print(f"Warning: Unknown sleep stage annotation: {desc_str}")
            
            if not stages:
                print("No valid sleep stages found in annotations")
            else:
                print(f"Successfully loaded {len(stages)} sleep stages")
            
            return np.array(stages)
            
        except Exception as e:
            print(f"Error loading hypnogram {hypno_file}: {str(e)}")
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
        # Store original shape to ensure we return the same shape
        original_shape = segment.shape
        
        # Gaussian noise
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.0001, 0.0005)
            segment += np.random.normal(0, noise_level, segment.shape)
        
        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            segment *= scale
        
        # Time warping (with shape preservation)
        if np.random.random() < 0.3:
            # Calculate warped length
            time_warp = np.random.uniform(0.9, 1.1)
            warped_len = int(segment.shape[0] * time_warp)
            
            if warped_len != segment.shape[0]:
                # Resample to warped length
                warped = signal.resample(segment, warped_len, axis=0)
                
                if warped_len > segment.shape[0]:
                    # If expanded, crop to original size
                    segment = warped[:segment.shape[0]]
                else:
                    # If compressed, pad to original size
                    pad_len = segment.shape[0] - warped_len
                    # Pad with reflection of the signal
                    segment = np.pad(warped, ((0, pad_len), (0, 0)), mode='reflect')
        
        # Verify and fix shape if needed
        if segment.shape != original_shape:
            print(f"Warning: Shape mismatch in augmentation. Original: {original_shape}, Got: {segment.shape}")
            # Ensure exact shape match by cropping or padding
            if segment.shape[0] > original_shape[0]:
                segment = segment[:original_shape[0]]
            elif segment.shape[0] < original_shape[0]:
                pad_len = original_shape[0] - segment.shape[0]
                segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='reflect')
        
        return segment