import os

import numpy as np


class DroidSLAMAdapter:
    """
    Adapter around the DROID-SLAM repo to provide pose tracking.
    This expects the official DROID-SLAM repository to be available locally.
    """

    def __init__(self, repo_path, device="cuda"):
        self.repo_path = repo_path
        self.device = device
        self._initialized = False

        if not os.path.isdir(repo_path):
            raise FileNotFoundError(
                f"DROID-SLAM repo not found at {repo_path}. "
                "Please clone https://github.com/princeton-vl/DROID-SLAM"
            )

        # Lazy import to avoid hard dependency when not used.
        import sys

        if repo_path not in sys.path:
            sys.path.append(repo_path)

        # DROID-SLAM uses its own API. We keep this minimal and defer details
        # to initialization to avoid import errors in environments lacking deps.
        from droid import Droid  # type: ignore

        self.Droid = Droid
        self.droid = None

    def initialize(self, image, depth=None, intrinsics=None):
        """
        Initialize the DROID tracker on the first frame.
        """
        if self._initialized:
            return
        self.droid = self.Droid(image, depth=depth, intrinsics=intrinsics, device=self.device)
        self._initialized = True

    def track(self, image, depth=None, intrinsics=None):
        """
        Track pose for a new frame. Returns (c2w, confidence).
        """
        if not self._initialized:
            self.initialize(image, depth=depth, intrinsics=intrinsics)

        # DROID-SLAM updates internal state and provides pose estimates.
        # The API returns a list of poses; we use the latest.
        self.droid.track(image, depth=depth, intrinsics=intrinsics)
        traj = self.droid.poses  # shape: [N, 4, 4]
        if traj is None or len(traj) == 0:
            return np.eye(4, dtype=np.float32), 0.0
        c2w = traj[-1].cpu().numpy()
        return c2w, 1.0

