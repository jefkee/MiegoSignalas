import numpy as np
import mne
from .eeg_processor import EEGProcessor
from .stage_analyzer import StageAnalyzer

class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.eeg_processor = EEGProcessor(config['preprocessing'])
        self.stage_analyzer = StageAnalyzer()
        
    def process_recording(self, psg_file, hypno_file=None):
        """Process a single PSG recording"""
        try:
            # Load EDF file
            raw = mne.io.read_raw_edf(psg_file, preload=True)
            
            # Process EEG data
            processed_data = self.eeg_processor.process(raw)
            
            # Load and process hypnogram if available
            stages = None
            if hypno_file:
                try:
                    stages = self.stage_analyzer.analyze_hypnogram(hypno_file)
                except Exception as e:
                    print(f"Warning: Could not process hypnogram: {e}")
            
            return processed_data, stages
            
        except Exception as e:
            print(f"Error processing recording: {str(e)}")
            raise 