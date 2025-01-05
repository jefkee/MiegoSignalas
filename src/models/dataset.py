import numpy as np
import mne
from src.utils.data_utils import find_matching_files
from src.config.settings import REQUIRED_CHANNELS, PREPROCESSING_PARAMS
import os

class Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.required_channels = REQUIRED_CHANNELS
        self.window_size = PREPROCESSING_PARAMS['window_size']
        self.sampling_rate = PREPROCESSING_PARAMS['sampling_rate']
        self.overlap = PREPROCESSING_PARAMS['overlap']
        self.lowpass = PREPROCESSING_PARAMS['lowpass']
        self.highpass = PREPROCESSING_PARAMS['highpass']
        
    def load_training_data(self, batch_size=10):
        """Load and preprocess PSG and hypnogram data for training in batches"""
        X_all = []
        y_all = []
        
        # Find matching PSG and hypnogram files
        matching_files = find_matching_files(self.data_dir)
        total_files = len(matching_files)
        
        print(f"\nFound {total_files} matching files")
        
        # Process files in batches
        for i in range(0, total_files, batch_size):
            batch_files = matching_files[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            
            X_batch, y_batch = self._process_file_batch(batch_files)
            
            if len(X_batch) > 0:
                X_all.extend(X_batch)
                y_all.extend(y_batch)
                
            # Clear memory
            del X_batch
            del y_batch
            
        if len(X_all) == 0:
            return np.array([]), np.array([])
            
        # Convert to numpy arrays and reshape for the model
        X = np.array(X_all)
        y = np.array(y_all)
        
        # Clear memory
        del X_all
        del y_all
        
        # Reshape to (samples, timesteps, channels)
        X = np.transpose(X, (0, 2, 1))
        
        print(f"\nLoaded {len(X)} segments with shape {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        return X, y
        
    def _process_file_batch(self, file_batch):
        """Process a batch of files"""
        X_batch = []
        y_batch = []
        
        for psg_file, hypno_file in file_batch:
            try:
                print(f"\nProcessing {os.path.basename(psg_file)}")
                
                # Load PSG data with timeout
                try:
                    raw = mne.io.read_raw_edf(psg_file, preload=True)
                except Exception as e:
                    print(f"Error loading {psg_file}: {str(e)}")
                    continue
                    
                # Check channels
                if not all(ch in raw.ch_names for ch in self.required_channels):
                    missing = [ch for ch in self.required_channels if ch not in raw.ch_names]
                    print(f"Skipping {psg_file}: Missing required channels: {missing}")
                    continue
                    
                # Select and reorder channels
                raw.pick_channels(self.required_channels)
                
                # Apply filters with timeout
                try:
                    raw.filter(self.highpass, self.lowpass, method='iir')
                except Exception as e:
                    print(f"Error filtering {psg_file}: {str(e)}")
                    continue
                
                # Get raw data
                data = raw.get_data()
                
                # Normalize each channel
                for i in range(data.shape[0]):
                    mean = np.mean(data[i])
                    std = np.std(data[i])
                    data[i] = (data[i] - mean) / (std + 1e-8)
                
                # Load hypnogram
                try:
                    annot = mne.read_annotations(hypno_file)
                except Exception as e:
                    print(f"Error loading hypnogram {hypno_file}: {str(e)}")
                    continue
                
                # Create segments and labels
                segments, labels = self._create_segments(data, annot, raw.info['sfreq'])
                
                if segments is not None and labels is not None:
                    X_batch.extend(segments)
                    y_batch.extend(labels)
                    print(f"Successfully processed {len(segments)} segments")
                    
                # Clear memory
                del raw
                del data
                del segments
                del labels
                
            except Exception as e:
                print(f"Error processing {psg_file}: {str(e)}")
                continue
                
        return X_batch, y_batch
        
    def _create_segments(self, data, annot, sfreq):
        """Create segments from raw data and annotations"""
        # Convert annotations to sleep stages
        stages = self._annotations_to_stages(annot)
        if stages is None:
            return None, None
            
        # Calculate samples per window
        window_samples = int(self.window_size * sfreq)
        overlap_samples = int(window_samples * self.overlap)
        
        segments = []
        labels = []
        
        # Segment data with overlap
        for start in range(0, data.shape[1] - window_samples + 1, window_samples - overlap_samples):
            end = start + window_samples
            segment = data[:, start:end]
            
            # Get label for this segment (majority vote)
            stage_start = int(start / sfreq)
            stage_end = int(end / sfreq)
            segment_stages = stages[stage_start:stage_end]
            
            if len(segment_stages) > 0:
                label = max(set(segment_stages), key=segment_stages.count)
                segments.append(segment)
                labels.append(label)
                
        return segments, labels
        
    def _annotations_to_stages(self, annot):
        """Convert annotations to sleep stage labels"""
        # Map annotation descriptions to stage numbers
        stage_map = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,  # Combine stage 3 and 4 as N3
            'Sleep stage R': 4
        }
        
        stages = []
        for onset, duration, description in zip(annot.onset, annot.duration, annot.description):
            if description in stage_map:
                stage = stage_map[description]
                # Convert time to 1-second epochs
                n_epochs = int(duration)
                stages.extend([stage] * n_epochs)
                
        return stages if stages else None