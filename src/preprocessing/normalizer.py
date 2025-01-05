import numpy as np

class Normalizer:
    def normalize_segment(self, segment):
        """Normalize a single segment"""
        # Z-score normalization
        mean = np.mean(segment)
        std = np.std(segment)
        return (segment - mean) / (std + 1e-8)
        
    def normalize_batch(self, segments):
        """Normalize a batch of segments"""
        normalized = []
        for segment in segments:
            normalized.append(self.normalize_segment(segment))
        return np.array(normalized)