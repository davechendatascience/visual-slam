import numpy as np

class LoopDetector:
    def __init__(self, K):
        self.K = K
        self.keyframes = []
        
    def add_keyframe(self, kf):
        self.keyframes.append(kf)
        
    def detect_loop(self, kf):
        """
        Stub loop detection.
        Returns None or dict with 'c2w_aligned'
        """
        # TODO: Implement real loop detection
        return None
