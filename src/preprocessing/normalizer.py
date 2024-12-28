import numpy as np
from sklearn.preprocessing import StandardScaler

class Normalizer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def normalize_segment(self, segment):
        """Normalize a single segment"""
        # Reshape to 2D for StandardScaler
        original_shape = segment.shape
        segment_2d = segment.reshape(-1, segment.shape[-1])
        
        # Fit and transform
        normalized = self.scaler.fit_transform(segment_2d)
        
        # Reshape back
        return normalized.reshape(original_shape)

    def normalize_batch(self, segments):
        """Normalize a batch of segments"""
        # Reshape to 2D
        original_shape = segments.shape
        segments_2d = segments.reshape(-1, segments.shape[-1])
        
        # Fit and transform
        normalized = self.scaler.fit_transform(segments_2d)
        
        # Reshape back
        return normalized.reshape(original_shape)