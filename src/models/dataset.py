import numpy as np
import mne
from src.utils.data_utils import find_matching_files
from src.config.settings import REQUIRED_CHANNELS, PREPROCESSING_PARAMS

class Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.required_channels = REQUIRED_CHANNELS
        self.window_size = PREPROCESSING_PARAMS['window_size']
        self.sampling_rate = PREPROCESSING_PARAMS['sampling_rate']
        self.overlap = PREPROCESSING_PARAMS['overlap']
        self.lowpass = PREPROCESSING_PARAMS['lowpass']
        self.highpass = PREPROCESSING_PARAMS['highpass']
        
    def load_training_data(self):
        """Load and preprocess PSG and hypnogram data for training"""
        X = []
        y = []
        
        # Find matching PSG and hypnogram files
        matching_files = find_matching_files(self.data_dir)
        
        for psg_file, hypno_file in matching_files:
            try:
                # Load PSG data
                raw = mne.io.read_raw_edf(psg_file, preload=True)
                
                # Print available channels
                print(f"\nAvailable channels in {psg_file}:")
                print(raw.ch_names)
                
                # Check channels
                if not all(ch in raw.ch_names for ch in self.required_channels):
                    missing = [ch for ch in self.required_channels if ch not in raw.ch_names]
                    print(f"Skipping {psg_file}: Missing required channels: {missing}")
                    continue
                    
                # Select and reorder channels
                raw.pick_channels(self.required_channels)
                
                # Apply filters
                raw.filter(self.highpass, self.lowpass, method='iir')
                
                # Get raw data
                data = raw.get_data()
                
                # Normalize each channel
                for i in range(data.shape[0]):
                    mean = np.mean(data[i])
                    std = np.std(data[i])
                    data[i] = (data[i] - mean) / (std + 1e-8)
                
                # Load hypnogram
                annot = mne.read_annotations(hypno_file)
                
                # Create segments and labels
                segments, labels = self._create_segments(data, annot, raw.info['sfreq'])
                
                if segments is not None and labels is not None:
                    X.extend(segments)
                    y.extend(labels)
                    
            except Exception as e:
                print(f"Error processing {psg_file}: {str(e)}")
                continue
                
        if len(X) == 0:
            return np.array([]), np.array([])
            
        # Convert to numpy arrays and reshape for the model
        X = np.array(X)
        y = np.array(y)
        
        # Reshape to (samples, timesteps, channels)
        X = np.transpose(X, (0, 2, 1))
        
        print(f"\nLoaded {len(X)} segments with shape {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        return X, y
        
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