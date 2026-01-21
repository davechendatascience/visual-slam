import numpy as np


class GaussianMapper:
    def __init__(self, gaussian_map, K, depth_range=(0.1, 8.0), subsample=4):
        self.map = gaussian_map
        self.K = K
        self.depth_min, self.depth_max = depth_range
        self.subsample = subsample

    def add_keyframe(self, keyframe):
        rgb = keyframe.rgb.astype(np.float32) / 255.0
        depth = keyframe.depth.astype(np.float32)
        H, W = depth.shape
        ys, xs = np.indices((H, W))
        ys = ys[:: self.subsample, :: self.subsample]
        xs = xs[:: self.subsample, :: self.subsample]
        z = depth[:: self.subsample, :: self.subsample]
        valid = (z > self.depth_min) & (z < self.depth_max)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        xyz_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb[:: self.subsample, :: self.subsample].reshape(-1, 3)
        mask = valid.reshape(-1)
        xyz_cam = xyz_cam[mask]
        colors = colors[mask]

        R = keyframe.c2w[:3, :3]
        t = keyframe.c2w[:3, 3]
        xyz_world = (R @ xyz_cam.T).T + t
        finite = np.isfinite(xyz_world).all(axis=1) & np.isfinite(colors).all(axis=1)
        if finite.sum() == 0:
            return
        xyz_world = xyz_world[finite]
        colors = np.clip(colors[finite], 0.0, 1.0)
        self.map.add_gaussians(xyz_world, colors)

