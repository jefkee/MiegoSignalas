import unittest
import mne
import numpy as np
from src.preprocessing.eeg_processor import EEGProcessor
from src.preprocessing.stage_analyzer import StageAnalyzer

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.eeg_processor = EEGProcessor()
        self.stage_analyzer = StageAnalyzer()
    
    def test_stage_mapping(self):
        self.assertEqual(self.stage_analyzer.STAGE_MAPPING['Sleep stage W'], 0)
        self.assertEqual(self.stage_analyzer.STAGE_MAPPING['Sleep stage R'], 4)
