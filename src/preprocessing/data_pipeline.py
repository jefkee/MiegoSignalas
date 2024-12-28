from src.preprocessing.eeg_processor import EEGProcessor
from src.preprocessing.stage_analyzer import StageAnalyzer
from src.utils.data_utils import find_edf_pairs, load_edf_file
import os
import mne

class DataPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.eeg_processor = EEGProcessor()
        self.stage_analyzer = StageAnalyzer()
    
    def process_dataset(self):
        """Complete data processing pipeline"""
        # Find PSG and hypnogram pairs
        psg_files, hypno_files = find_edf_pairs(self.data_dir)
        
        processed_data = []
        labels = []
        
        for psg, hypno in zip(psg_files, hypno_files):
            # Load and process EEG
            raw_eeg = load_edf_file(os.path.join(self.data_dir, psg))
            if raw_eeg is not None:
                processed_eeg = self.eeg_processor.preprocess_signal(raw_eeg)
                processed_data.append(processed_eeg)
                
                # Load and process hypnogram
                annot = mne.read_annotations(os.path.join(self.data_dir, hypno))
                stage_labels = self.stage_analyzer.analyze_hypnogram(annot)
                labels.append(stage_labels)
        
        return processed_data, labels 

    def preprocess_signal(self, data):
        """Enhanced signal preprocessing"""
        # Bandpass filter
        data = mne.filter.filter_data(
            data, sfreq=100, 
            l_freq=0.3, h_freq=35,  # Adjusted frequency bands
            method='iir',
            iir_params={'order': 4, 'ftype': 'butter'}
        )
        
        # Remove artifacts
        data = mne.preprocessing.ICA(
            n_components=0.99,
            random_state=42
        ).fit_transform(data)
        
        return data 