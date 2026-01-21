from dataclasses import dataclass
import numpy as np


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
    keypoints: list
    descriptors: np.ndarray

