import numpy as np
import torch

class GaussianMapper:
    def __init__(self, gaussian_map, K, subsample=1):
        self.gaussian_map = gaussian_map
        self.K = K
        self.subsample = subsample

    def add_keyframe(self, kf):
        # kf has rgb, depth, c2w
        rgb = kf.rgb
        depth = kf.depth
        c2w = kf.c2w
        
        # Subsample
        if self.subsample > 1:
            rgb = rgb[::self.subsample, ::self.subsample]
            depth = depth[::self.subsample, ::self.subsample]
            
        H, W = depth.shape
        K = self.K.copy()
        K[:2] /= self.subsample

        # Backprojection
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        x = x.flatten()
        y = y.flatten()
        z = depth.flatten()
        
        valid = (z > 0.1) & (z < 10.0)
        x = x[valid]
        y = y[valid]
        z = z[valid]
        rgbs = rgb.reshape(-1, 3)[valid] / 255.0
        
        if len(z) == 0:
            return
            
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        Z = z
        
        points_cam = np.stack([X, Y, Z], axis=1)
        
        # Transform to world
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        points_world = (R @ points_cam.T).T + t
        
        self.gaussian_map.add_gaussians(points_world, rgbs)
