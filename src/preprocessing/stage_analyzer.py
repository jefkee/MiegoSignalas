import mne
import numpy as np
from typing import Dict

class StageAnalyzer:
    """
    Analyzes sleep stages from hypnogram data
    """
    # Sleep stage mapping according to AASM standards
    STAGE_MAPPING = {
        'Sleep stage W': 0,    # Wake
        'Sleep stage 1': 1,    # N1
        'Sleep stage 2': 2,    # N2
        'Sleep stage 3': 3,    # N3 (combining 3 & 4)
        'Sleep stage 4': 3,    # N3 (combining 3 & 4)
        'Sleep stage R': 4,    # REM
        'Movement time': -1,   # Will be excluded
        'Sleep stage ?': -1    # Will be excluded
    }

    def analyze_hypnogram(self, annotations: mne.Annotations) -> Dict:
        """
        Analyze sleep stage distribution
        """
        stats = {}
        for annot in annotations:
            if annot['description'] in self.STAGE_MAPPING:
                stage = annot['description']
                if stage not in stats:
                    stats[stage] = 0
                stats[stage] += 1
        return stats
