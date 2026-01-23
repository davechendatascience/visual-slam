from dataclasses import dataclass
import numpy as np
from typing import Optional, Any

@dataclass
class Frame:
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    c2w: np.ndarray

@dataclass
class Keyframe:
    keyframe_id: int
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    c2w: np.ndarray
    keypoints: Optional[list] = None
    descriptors: Optional[np.ndarray] = None
