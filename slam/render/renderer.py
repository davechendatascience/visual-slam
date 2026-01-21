import numpy as np
import torch


def render_gaussians(
    means,
    colors,
    opacities,
    scales,
    w2c,
    K,
    height,
    width,
    max_points=20000,
):
    """
    Minimal CPU renderer for MVP visualization and photometric reference.
    This is not a full splatting renderer but approximates Gaussian splats
    using a small 2D kernel. Intended for correctness over speed.
    """
    if isinstance(means, torch.Tensor):
        means = means.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    if isinstance(opacities, torch.Tensor):
        opacities = opacities.detach().cpu().numpy()
    if isinstance(scales, torch.Tensor):
        scales = scales.detach().cpu().numpy()

    if means.shape[0] == 0:
        return np.zeros((height, width, 3), dtype=np.float32)

    if means.shape[0] > max_points:
        idx = np.random.choice(means.shape[0], max_points, replace=False)
        means = means[idx]
        colors = colors[idx]
        opacities = opacities[idx]
        scales = scales[idx]

    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts = (R @ means.T).T + t
    z = pts[:, 2]
    mask = z > 0.1
    pts = pts[mask]
    z = z[mask]
    colors = colors[mask]
    opacities = opacities[mask]
    scales = scales[mask]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (pts[:, 0] * fx / z + cx).astype(np.int32)
    y = (pts[:, 1] * fy / z + cy).astype(np.int32)

    img = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)

    for i in range(pts.shape[0]):
        if x[i] < 0 or x[i] >= width or y[i] < 0 or y[i] >= height:
            continue
        sigma = max(1.0, scales[i].mean() * 50.0)
        radius = int(max(1, sigma * 2))
        x0 = max(0, x[i] - radius)
        x1 = min(width - 1, x[i] + radius)
        y0 = max(0, y[i] - radius)
        y1 = min(height - 1, y[i] + radius)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                dx = xx - x[i]
                dy = yy - y[i]
                w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
                w = w * float(opacities[i])
                img[yy, xx, :] += colors[i] * w
                weight[yy, xx, 0] += w

    mask = weight > 1e-6
    img[mask.squeeze()] /= weight[mask].reshape(-1, 1)
    return img.clip(0.0, 1.0)

