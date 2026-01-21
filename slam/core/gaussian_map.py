import numpy as np
import torch


class GaussianMap:
    def __init__(self, device="cuda"):
        self.device = device
        self.means = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.scales = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.colors = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.opacities = torch.empty((0, 1), dtype=torch.float32, device=device)

    def add_gaussians(self, means, colors, scale=0.02, opacity=0.8):
        if means.shape[0] == 0:
            return
        if isinstance(means, np.ndarray):
            means = torch.tensor(means, dtype=torch.float32, device=self.device)
        if isinstance(colors, np.ndarray):
            colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        means = means.to(self.device, dtype=torch.float32)
        colors = colors.to(self.device, dtype=torch.float32)
        mask = torch.isfinite(means).all(dim=1) & torch.isfinite(colors).all(dim=1)
        if mask.sum() == 0:
            return
        means = means[mask]
        colors = colors[mask].clamp(0.0, 1.0)
        scales = torch.ones((means.shape[0], 3), dtype=torch.float32, device=self.device) * scale
        opacities = torch.ones((means.shape[0], 1), dtype=torch.float32, device=self.device) * opacity
        self.means = torch.cat([self.means, means], dim=0)
        self.scales = torch.cat([self.scales, scales], dim=0)
        self.colors = torch.cat([self.colors, colors], dim=0)
        self.opacities = torch.cat([self.opacities, opacities], dim=0)

    def prune(self, min_opacity=0.05):
        mask = self.opacities.squeeze() > min_opacity
        if mask.numel() == 0 or mask.sum() == 0:
            return
        self.means = self.means[mask]
        self.scales = self.scales[mask]
        self.colors = self.colors[mask]
        self.opacities = self.opacities[mask]

    def apply_global_transform(self, T_world_new_from_world_old):
        R = T_world_new_from_world_old[:3, :3]
        t = T_world_new_from_world_old[:3, 3]
        means = self.means.detach().cpu().numpy()
        means = (R @ means.T).T + t
        self.means = torch.tensor(means, dtype=torch.float32, device=self.device)

    def to_ply(self, path):
        means = self.means.detach().cpu().numpy()
        colors = (self.colors.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {means.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for i in range(means.shape[0]):
                x, y, z = means[i]
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

