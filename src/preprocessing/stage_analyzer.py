import mne
import numpy as np

class StageAnalyzer:
    def __init__(self):
        self.stage_map = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,  # Combine stage 3 and 4
            'Sleep stage R': 4
        }
        
    def analyze_hypnogram(self, hypno_file):
        """Analyze hypnogram file and extract sleep stages"""
        try:
            # Load annotations
            annotations = mne.read_annotations(hypno_file)
            
            # Extract stages
            stages = []
            for description in annotations.description:
                if description in self.stage_map:
                    stages.append(self.stage_map[description])
                    
            return np.array(stages)
            
        except Exception as e:
            print(f"Error analyzing hypnogram: {str(e)}")
            raise
