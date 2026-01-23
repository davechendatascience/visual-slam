import numpy as np
import torch
import torch.nn.functional as F


def _rodrigues(rotvec):
    theta = torch.norm(rotvec)
    if theta < 1e-8:
        return torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    k = rotvec / theta
    K = torch.tensor(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
        device=rotvec.device,
        dtype=rotvec.dtype,
    )
    R = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    R = R + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


class PhotometricTracker:
    def __init__(self, K, device="cuda", max_points=8000, iters=20, lr=1e-2):
        self.K = torch.tensor(K, dtype=torch.float32, device=device)
        self.device = device
        self.max_points = max_points
        self.iters = iters
        self.lr = lr

    def track(self, gaussian_map, init_c2w, rgb, depth=None):
        if gaussian_map.means.shape[0] == 0:
            return init_c2w, 0.0

        rgb_t = torch.tensor(rgb, dtype=torch.float32, device=self.device) / 255.0
        H, W, _ = rgb_t.shape

        if depth is not None:
             depth_t = torch.tensor(depth, dtype=torch.float32, device=self.device)
        
        means = gaussian_map.means
        colors = gaussian_map.colors

        if means.shape[0] > self.max_points:
            idx = torch.randperm(means.shape[0], device=self.device)[: self.max_points]
            means = means[idx]
            colors = colors[idx]

        init_c2w_t = torch.tensor(init_c2w, dtype=torch.float32, device=self.device)
        rot_delta = torch.zeros(3, device=self.device, requires_grad=True)
        trans_delta = torch.zeros(3, device=self.device, requires_grad=True)

        optim = torch.optim.Adam([rot_delta, trans_delta], lr=self.lr)

        for _ in range(self.iters):
            optim.zero_grad()

            R_delta = _rodrigues(rot_delta)
            c2w = init_c2w_t.clone()
            c2w[:3, :3] = R_delta @ c2w[:3, :3]
            c2w[:3, 3] = c2w[:3, 3] + trans_delta

            w2c = torch.inverse(c2w)
            pts = (w2c[:3, :3] @ means.T).T + w2c[:3, 3]
            z = pts[:, 2]
            valid = z > 0.1
            pts = pts[valid]
            cols = colors[valid]
            z = z[valid]
            if pts.shape[0] < 100:
                return init_c2w, 0.0

            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            u = (pts[:, 0] * fx / z + cx)
            v = (pts[:, 1] * fy / z + cy)
            finite = torch.isfinite(u) & torch.isfinite(v)
            u = u[finite]
            v = v[finite]
            cols = cols[finite]
            # Must filter z properly here too since it needs to align with others
            z = z[finite] 
            
            in_bounds = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
            u = u[in_bounds]
            v = v[in_bounds]
            cols = cols[in_bounds]
            z = z[in_bounds]
            
            if u.numel() < 100:
                return init_c2w, 0.0

            # Normalize to [-1, 1] for grid_sample
            u_norm = (u / (W - 1)) * 2.0 - 1.0
            v_norm = (v / (H - 1)) * 2.0 - 1.0
            u_norm = u_norm.clamp(-1.0, 1.0)
            v_norm = v_norm.clamp(-1.0, 1.0)
            grid = torch.stack([u_norm, v_norm], dim=-1).view(1, -1, 1, 2)

            # Sample RGB
            img = rgb_t.permute(2, 0, 1).unsqueeze(0)
            sampled = F.grid_sample(img, grid, mode="bilinear", align_corners=True)
            gt_rgb = sampled.squeeze(0).squeeze(-1).permute(1, 0)

            rgb_loss = torch.abs(cols - gt_rgb).mean()
            
            depth_loss = 0.0
            if depth is not None:
                # Sample Depth
                # Add channel dim for grid_sample
                dimg = depth_t.unsqueeze(0).unsqueeze(0) 
                sampled_d = F.grid_sample(dimg, grid, mode="nearest", align_corners=True)
                gt_depth = sampled_d.squeeze()
                
                # Check for valid depth in GT
                valid_depth = (gt_depth > 0.1) & (gt_depth < 10.0) & torch.isfinite(gt_depth)
                
                if valid_depth.sum() > 10:
                     depth_loss = torch.abs(z[valid_depth] - gt_depth[valid_depth]).mean()

            loss = rgb_loss + 0.5 * depth_loss
            loss.backward()
            optim.step()

        c2w_final = c2w.detach().cpu().numpy()
        return c2w_final, float(in_bounds.float().mean().item())
