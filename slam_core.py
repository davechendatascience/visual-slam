import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import plotly.graph_objects as go

# Optional CUDA Rasterizer
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    CUDA_RASTERIZER_AVAILABLE = True
except ImportError:
    CUDA_RASTERIZER_AVAILABLE = False


# --- 1. DATASET LOADER ---
def associate_data(root_dir):
    def read_file_list(filename):
        file = open(filename)
        data = file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if len(line) > 0 and line[0] != "#"]
        list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
        return dict(list)

    if not os.path.exists(os.path.join(root_dir, "rgb.txt")):
         return []
         
    rgb_list = read_file_list(os.path.join(root_dir, "rgb.txt"))
    gt_list = read_file_list(os.path.join(root_dir, "groundtruth.txt"))
    def_list = read_file_list(os.path.join(root_dir, "depth.txt"))
    
    rgb_timestamps = sorted(rgb_list.keys())
    gt_timestamps = sorted(gt_list.keys())
    dep_timestamps = sorted(def_list.keys())
    
    matches = []
    max_diff = 0.02
    
    for t in rgb_timestamps:
        best_gt = min(gt_timestamps, key=lambda x: abs(x - t))
        best_dep = min(dep_timestamps, key=lambda x: abs(x - t))
        
        if abs(best_gt - t) < max_diff and abs(best_dep - t) < max_diff:
            matches.append((t, best_gt, best_dep))
            
    data = []
    for t_rgb, t_gt, t_dep in matches:
        rgb_f = rgb_list[t_rgb][0]
        unidepth = os.path.join(root_dir, "depth_unidepth", os.path.basename(rgb_f).replace(".png", ".npy"))
        sensor = os.path.join(root_dir, def_list[t_dep][0])
        
        gt = gt_list[t_gt]
        tx, ty, tz = float(gt[0]), float(gt[1]), float(gt[2])
        qx, qy, qz, qw = float(gt[3]), float(gt[4]), float(gt[5]), float(gt[6])
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w = np.eye(4); c2w[:3, :3] = rot; c2w[:3, 3] = [tx, ty, tz]
        
        data.append({
            "rgb_path": os.path.join(root_dir, rgb_f),
            "unidepth_path": unidepth,
            "sensor_depth_path": sensor,
            "c2w": c2w,
            "timestamp": t_rgb
        })
    return data

# --- 2. RASTERIZERS (Robust) ---

def rasterize_soft(means, colors, opacities, scales, quats, viewmat, K, height, width):
    device = means.device
    R = viewmat[:3, :3]; t = viewmat[:3, 3]
    means_c = (R @ means.T).T + t
    
    # Filter Z > 0.1 to avoid singularities
    mask = means_c[:, 2] > 0.1
    points = means_c[mask]; colors = colors[mask]
    if points.shape[0] == 0: return torch.zeros((height, width, 4), device=device)
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    z = points[:, 2]
    x = points[:, 0] * fx / z + cx
    y = points[:, 1] * fy / z + cy 
    
    # Strict Screen Bounds for Bilinear
    mask_screen = (x >= 0) & (x < width - 1.0) & (y >= 0) & (y < height - 1.0)
    points = points[mask_screen]; colors = colors[mask_screen]
    x = x[mask_screen]; y = y[mask_screen]; z = z[mask_screen]
    if points.shape[0] == 0: return torch.zeros((height, width, 4), device=device)

    # Bilinear Weights
    x0 = torch.floor(x).long(); y0 = torch.floor(y).long()
    x1 = x0 + 1; y1 = y0 + 1
    
    dx = x - x0; dy = y - y0
    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy
    
    # Indices
    idx_a = y0 * width + x0
    idx_b = y0 * width + x1
    idx_c = y1 * width + x0
    idx_d = y1 * width + x1
    
    # Buffer: RGB (3) + Depth (1) + Weight (1) = 5
    buffer = torch.zeros((height * width, 5), device=device)
    
    # Accumulate
    # We stack the 4 corners
    all_indices = torch.cat([idx_a, idx_b, idx_c, idx_d], dim=0)
    all_weights = torch.cat([wa, wb, wc, wd], dim=0).unsqueeze(1)
    
    # Features: [R, G, B, Z]
    feat = torch.cat([colors, z.unsqueeze(1)], dim=1)
    all_feat = feat.repeat(4, 1)
    
    # Atomic Add (safe on cuda/GPU)
    # Val = Feature * Weight
    vals = torch.cat([all_feat * all_weights, all_weights], dim=1)
    buffer.index_add_(0, all_indices, vals)
    
    # Normalize
    total_weight = buffer[:, 4:5]
    # Avoid div by zero artifacts (if weight is tiny, we might get weirdness, but usually fine)
    mask_valid = total_weight > 1e-4
    
    final_feat = torch.zeros_like(buffer[:, 0:4])
    final_feat[mask_valid.squeeze()] = buffer[mask_valid.squeeze(), 0:4] / total_weight[mask_valid.squeeze()]
    
    # Clamp RGB to [0, 1] to prevent "White Pixel" explosion
    rgb = torch.clamp(final_feat[:, 0:3], 0.0, 1.0)
    depth = final_feat[:, 3:4]
    
    return torch.cat([rgb, depth], dim=1).reshape(height, width, 4)

def rasterize_solid(means, colors, opacities, scales, quats, viewmat, K, height, width):
    device = means.device
    R = viewmat[:3, :3]; t = viewmat[:3, 3]
    means_c = (R @ means.T).T + t
    mask = means_c[:, 2] > 0.1
    points = means_c[mask]; colors = colors[mask]
    if points.shape[0] == 0: return torch.zeros((height, width, 4), device=device)
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    z = points[:, 2]
    
    # Standard Pinhole Projection
    x = points[:, 0] * fx / z + cx
    y = points[:, 1] * fy / z + cy
    
    mask_screen = (x >= 0) & (x < width - 1) & (y >= 0) & (y < height - 1)
    points = points[mask_screen]; colors = colors[mask_screen]
    x = x[mask_screen]; y = y[mask_screen]; z = z[mask_screen]
    if points.shape[0] == 0: return torch.zeros((height, width, 4), device=device)

    # Z-Buffer Sort (Painter's Algo)
    sorted_indices = torch.argsort(z, descending=True)
    x = x[sorted_indices]; y = y[sorted_indices]; z = z[sorted_indices]
    points = points[sorted_indices]; colors = colors[sorted_indices]

    # Z-Buffer Fill
    ix = torch.floor(x).long(); iy = torch.floor(y).long()
    pix_idx = iy * width + ix
    z_buffer = torch.ones(height * width, device=device) * 100.0
    z_buffer[pix_idx] = z
    
    # Visibility Filter
    orig_ix = torch.floor(x).long(); orig_iy = torch.floor(y).long()
    orig_idx = orig_iy * width + orig_ix
    min_z = z_buffer[orig_idx]
    is_visible = z <= (min_z + 0.05)
    
    points = points[is_visible]; colors = colors[is_visible]
    x = x[is_visible]; y = y[is_visible]; z = z[is_visible]
    
    if points.shape[0] == 0: return torch.zeros((height, width, 4), device=device)

    # Splat Visible
    x0 = torch.floor(x).long(); y0 = torch.floor(y).long()
    x1 = torch.clamp(x0 + 1, 0, width-1); y1 = torch.clamp(y0 + 1, 0, height-1)
    
    dx = x - x0; dy = y - y0
    wa = (1-dx)*(1-dy); wb = dx*(1-dy); wc = (1-dx)*dy; wd = dx*dy
    
    idx_a = y0 * width + x0; idx_b = y1 * width + x0
    idx_c = y0 * width + x1; idx_d = y1 * width + x1
    
    buffer = torch.zeros((height * width, 5), device=device)
    
    all_indices = torch.cat([idx_a, idx_b, idx_c, idx_d], dim=0)
    all_weights = torch.cat([wa, wb, wc, wd], dim=0).unsqueeze(1)
    
    feat = torch.cat([colors, z.unsqueeze(1)], dim=1)
    all_feat = feat.repeat(4, 1)
    
    buffer.index_add_(0, all_indices, torch.cat([all_feat * all_weights, all_weights], dim=1))
    
    total_weight = buffer[:, 4:5] + 1e-6
    final_feat = buffer[:, 0:4] / total_weight
    
    rgb = torch.clamp(final_feat[:, 0:3], 0, 1)
    depth = final_feat[:, 3:4]
    return torch.cat([rgb, depth], dim=1).reshape(height, width, 4)

def pure_pytorch_rasterization(means, colors, opacities, scales, quats, viewmat, K, height, width, mode='solid'):
    if mode == 'soft':
        return rasterize_soft(means, colors, opacities, scales, quats, viewmat, K, height, width)
    else:
        return rasterize_solid(means, colors, opacities, scales, quats, viewmat, K, height, width)

def get_projection_matrix(K, H, W, n=0.01, f=100.0):
    fovx = 2 * np.arctan(W / (2 * K[0, 0]))
    fovy = 2 * np.arctan(H / (2 * K[1, 1]))
    
    tan_fovx = np.tan(fovx / 2)
    tan_fovy = np.tan(fovy / 2)
    
    top = tan_fovy * n
    bottom = -top
    right = tan_fovx * n
    left = -right
    
    P = torch.zeros(4, 4)
    z_sign = 1.0 # OpenGL convention usually? Or different for diff-gaussian? 
    # diff-gaussian expects standard OpenGL projection
    
    P[0, 0] = 2 * n / (right - left)
    P[1, 1] = 2 * n / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * f / (f - n)
    P[2, 3] = -(f * n) / (f - n)
    
    return P, np.tan(fovx/2), np.tan(fovy/2)


# --- 3. MODEL (Device Agnostic) ---
class SimpleGaussianModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.means = nn.Parameter(torch.empty(0, 3, device=device))
        self.scales = nn.Parameter(torch.empty(0, 3, device=device))
        self.quats = nn.Parameter(torch.empty(0, 4, device=device))
        self.opacities = nn.Parameter(torch.empty(0, 1, device=device))
        self.colors = nn.Parameter(torch.empty(0, 3, device=device))
        self.optimizer = None

    def training_setup(self, lr_means=0.00016, lr_colors=0.0025, lr_opacities=0.05, lr_scales=0.005, lr_quats=0.001):
        if self.optimizer: return
        
        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': lr_means},
            {'params': [self.colors], 'lr': lr_colors},
            {'params': [self.opacities], 'lr': lr_opacities},
            {'params': [self.scales], 'lr': lr_scales},
            {'params': [self.quats], 'lr': lr_quats}
        ], lr=0.0)

        
    def add_gaussians(self, new_means, new_colors):
        with torch.no_grad():
            N_new = new_means.shape[0]
            if N_new == 0: return
            new_scales = torch.ones(N_new, 3, device=self.device) * -5.0 
            new_quats = torch.zeros(N_new, 4, device=self.device); new_quats[:, 0] = 1.0
            new_opacities = torch.zeros(N_new, 1, device=self.device)
            new_colors = torch.clamp(new_colors, 0.01, 0.99)
            new_colors = torch.logit(new_colors) 
            self.means = nn.Parameter(torch.cat([self.means, new_means]))
            self.scales = nn.Parameter(torch.cat([self.scales, new_scales]))
            self.quats = nn.Parameter(torch.cat([self.quats, new_quats]))
            self.opacities = nn.Parameter(torch.cat([self.opacities, new_opacities]))
            self.colors = nn.Parameter(torch.cat([self.colors, new_colors]))
            
        self.optimizer = None # Invalidate optimizer (shapes changed)


    def forward(self, viewmat, K, height, width, mode='solid'):
        render_colors = torch.sigmoid(self.colors)
        opacities = torch.sigmoid(self.opacities)
        
        # Check for CUDA Rasterizer
        if CUDA_RASTERIZER_AVAILABLE and self.device != 'cpu':
             # Prepare Inputs for diff-gaussian-rasterization
             # 1. Proj Matrix
             # Note: simple K-based projection
             fx = K[0,0]; fy = K[1,1]
             fovx = 2 * torch.atan(width / (2 * fx))
             fovy = 2 * torch.atan(height / (2 * fy))
             tanfovx = torch.tan(fovx * 0.5)
             tanfovy = torch.tan(fovy * 0.5)
             
             # Create OpenGL Projection Matrix
             zfar = 100.0; znear = 0.01
             P = torch.zeros(4, 4, device=self.device)
             P[0, 0] = 1.0 / tanfovx
             P[1, 1] = 1.0 / tanfovy
             P[2, 2] = zfar / (zfar - znear)
             P[2, 3] = -(zfar * znear) / (zfar - znear)
             P[3, 2] = 1.0
             
             # World to Screen = P @ View
             # diff-gaussian expects "full_proj_transform" -> P @ View
             # But View is W2C.
             
             # Transpose check: diff-gaussian often expects Row-Major or specific layout?
             # Standard: full_proj = Proj @ View
             
             # NOTE: diff-gaussian documentation requires Transposed matrices if inputs are Row-Major?
             # Let's assume standard PyTorch Layout (Row Major) but math is Column Major?
             # The codebase typically uses transposed ViewMat.
             
             viewmat_t = viewmat.transpose(0, 1) # to Column Major?
             proj_t = P.transpose(0, 1)
             full_proj = (P @ viewmat).transpose(0, 1)
             
             camera_center = torch.inverse(viewmat)[:3, 3]
             
             raster_settings = GaussianRasterizationSettings(
                image_height=int(height),
                image_width=int(width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32),
                scale_modifier=1.0,
                viewmatrix=viewmat_t,
                projmatrix=full_proj,
                sh_degree=0,
                campos=camera_center,
                prefiltered=False,
                debug=False
             )
             
             rasterizer = GaussianRasterizer(raster_settings)
             
             rendered_image, radii = rasterizer(
                means3D = self.means,
                means2D = torch.zeros_like(self.means),
                shs = None,
                colors_precomp = render_colors, # RGB
                opacities = opacities,
                scales = torch.exp(self.scales),
                rotations = self.quats,
                cov3D_precomp = None
             )
             
             # Diff-Gaussian returns (3, H, W). We need (H, W, 4) with Depth?
             # Wait, diff-gaussian doesn't return depth by default in RGB mode.
             rgb = rendered_image.permute(1, 2, 0) # (H, W, 3)
             
             # Hack for depth: use distance to mean? No, that's not rasterized depth.
             # If we need depth for tracking, we might need a modified rasterizer or just use RGB loss.
             
             # If strictly RGB mode, return dummy depth
             return torch.cat([rgb, torch.zeros(height, width, 1, device=self.device)], dim=2)

        return pure_pytorch_rasterization(
            self.means, render_colors, opacities, torch.exp(self.scales), self.quats,
            viewmat, K, height, width, mode=mode
        )

        
    def prune_points(self, min_opacity=0.005):
        # Prune points with low opacity to keep map sparse
        with torch.no_grad():
            # Get valid mask
            valid_mask = torch.sigmoid(self.opacities).squeeze() > min_opacity
            if valid_mask.sum() == 0: return # Safety 
            
            # Keep only valid
            self.means = nn.Parameter(self.means[valid_mask])
            self.scales = nn.Parameter(self.scales[valid_mask])
            self.quats = nn.Parameter(self.quats[valid_mask])
            self.opacities = nn.Parameter(self.opacities[valid_mask])
            self.colors = nn.Parameter(self.colors[valid_mask])
            
        self.optimizer = None # Invalidate optimizer (shapes changed)



# --- 4. TRACKING & MAPPING HELPER ---

class GeometricTracker:
    def __init__(self, K, H, W):
        self.K = K 
        self.H, self.W = H, W
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx, self.cy = K[0,2], K[1,2]
        
    def track(self, prev_img, prev_depth, curr_img, prev_c2w):
        '''
        Fast RGB-D Tracking using Optical Flow + PnP
        Inputs:
            prev_img, curr_img: numpy arrays (H, W, 3) or (H, W) - will convert to gray
            prev_depth: (H, W) metric depth
            prev_c2w: (4, 4) numpy pose
        Returns:
            curr_c2w: Estimated pose
        '''
        # 1. Prepare Images
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY) if prev_img.ndim==3 else prev_img
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if curr_img.ndim==3 else curr_img
        
        # 2. Extract Features (Shi-Tomasi / GFTT)
        # We need points that have DEPTH
        feats_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
        
        if feats_prev is None or len(feats_prev) < 5:
            # Tracking Failed (Too few features)
            return prev_c2w # Decay to constant velocity (or static)
            
        feats_prev = feats_prev.squeeze() # (N, 2)
        
        # 3. Filter by Depth
        valid_indices = []
        points_3d = []
        points_2d_prev = []
        
        for i, pt in enumerate(feats_prev):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < self.H and 0 <= x < self.W:
                z = prev_depth[y, x]
                if 0.1 < z < 8.0:
                    # Backproject to 3D (Camera Frame)
                    X = (pt[0] - self.cx) * z / self.fx
                    Y = (pt[1] - self.cy) * z / self.fy
                    points_3d.append([X, Y, z])
                    points_2d_prev.append(pt)
                    
        points_3d = np.array(points_3d) # (N, 3)
        points_2d_prev = np.array(points_2d_prev, dtype=np.float32) # (N, 2)
        
        if len(points_3d) < 8: return prev_c2w # Need points for PnP
        
        # 4. Optical Flow (KLT)
        points_2d_curr, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points_2d_prev, None, winSize=(21, 21))
        
        # Filter good tracks
        good_3d = points_3d[status.flatten()==1]
        good_2d = points_2d_curr[status.flatten()==1]
        
        if len(good_3d) < 8: return prev_c2w
        
        # 5. Solve PnP (RANSAC)
        # We solve for Pose of Current Camera relative to PREVIOUS Frame's 3D points
        # But wait, the 3D points are in Prev Camera Frame.
        # So PnP gives T_curr_prev (Pose of Prev defined in Curr frame... or w2c?)
        # cv2.solvePnP gives rvec, tvec that maps World (3D) to Camera (2D).
        # Here "World" is Prev Camera Frame.
        # So we get T_curr_prev.
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(good_3d, good_2d, self.K, None, iterationsCount=100, reprojectionError=2.0)
        
        if not success: return prev_c2w
        
        R, _ = cv2.Rodrigues(rvec)
        T = tvec
        
        # T_curr_prev (Transform from Prev to Curr)
        # T_curr_world = T_curr_prev * T_prev_world
        # w2c_curr = T_curr_prev * w2c_prev ?? NO
        
        # Let's align: 
        # X_curr = R * X_prev + T
        # X_world = c2w_prev * X_prev  => X_prev = w2c_prev * X_world
        # X_curr = R * (w2c_prev * X_world) + T 
        # X_curr = (R * w2c_prev) * X_world + T
        # So w2c_curr = [ R*R_prev  |  R*t_prev + T ] ?? 
        
        # Easier: Construction full 4x4 matrix
        T_curr_prev = np.eye(4)
        T_curr_prev[:3, :3] = R
        T_curr_prev[:3, 3] = T.squeeze()
        
        w2c_prev = np.linalg.inv(prev_c2w)
        w2c_curr = T_curr_prev @ w2c_prev
        
        c2w_curr = np.linalg.inv(w2c_curr)
        
        # Stability check (prevent jumps)
        delta = np.linalg.norm(c2w_curr[:3, 3] - prev_c2w[:3, 3])
        if delta > 0.5: return prev_c2w # Large jump = Fail
        
        return c2w_curr

def optimize_tracking(model, init_c2w, gt_rgb, K, H, W, iters=50, gt_depth=None):

    """
    Optimized Tracking with Coarse-to-Fine Pyramid using ORIGINAL Rasterizer.
    
    Strategy:
    - Stage 1 (Coarse): Run 60% of iterations at 1/4 resolution (1/16th pixels).
      This converges large motions extremely fast.
    - Stage 2 (Fine): Run 40% of iterations at full resolution for final precision.
    """
    device = model.means.device
    
    # --- 1. SETUP PARAMETERS ---
    c2w_init_torch = torch.tensor(init_c2w, dtype=torch.float32, device=device)
    w2c_init = torch.inverse(c2w_init_torch)
    R_init = w2c_init[:3, :3]
    T_init = w2c_init[:3, 3]
    
    # Trainable params: Delta update
    trans_param = nn.Parameter(T_init.clone())
    rot_delta_param = nn.Parameter(torch.zeros(3, device=device))
    
    # Use Adam. eps=1e-8 is standard, but 1e-15 can help stability if gradients are tiny
    optimizer_track = torch.optim.Adam([trans_param, rot_delta_param], lr=0.005)
    
    # --- 2. PRE-COMPUTE PYRAMID LEVELS ---
    # Downscale factor
    scale = 0.25 
    H_small, W_small = int(H * scale), int(W * scale)
    
    # 2a. Downscale Ground Truth RGB
    # Input: [H, W, 3] -> Permute to [1, 3, H, W] for interpolate -> Back to [H, W, 3]
    gt_rgb_small = F.interpolate(
        gt_rgb.permute(2,0,1).unsqueeze(0), 
        size=(H_small, W_small), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0).permute(1,2,0)
    
    # 2b. Scale Intrinsics (K)
    K_small = K.clone()
    K_small[:2, :] *= scale # Scale fx, fy, cx, cy
    
    # 2c. Downscale Depth (if present)
    has_depth = gt_depth is not None
    if has_depth:
        if isinstance(gt_depth, np.ndarray):
            gt_depth = torch.tensor(gt_depth, dtype=torch.float32, device=device)
        
        gt_depth_flat = gt_depth.reshape(-1)
        valid_depth_mask = (gt_depth_flat > 0.1) & (gt_depth_flat < 8.0)
        
        # Nearest neighbor for depth to preserve sharp edges
        gt_depth_small = F.interpolate(
            gt_depth.unsqueeze(0).unsqueeze(0),
            size=(H_small, W_small),
            mode='nearest'
        ).squeeze()
        gt_depth_small_flat = gt_depth_small.reshape(-1)
        valid_depth_small_mask = (gt_depth_small_flat > 0.1) & (gt_depth_small_flat < 8.0)

    # --- 3. OPTIMIZATION LOOP ---
    coarse_iters = int(iters * 0.6)
    
    for i in range(iters):
        optimizer_track.zero_grad()
        
        # --- SWITCH STAGES ---
        if i < coarse_iters:
            # COARSE STAGE (Fast)
            cur_H, cur_W = H_small, W_small
            cur_K = K_small
            cur_gt_rgb = gt_rgb_small
            if has_depth:
                cur_gt_depth_flat = gt_depth_small_flat
                cur_valid_mask = valid_depth_small_mask
        else:
            # FINE STAGE (Precise)
            cur_H, cur_W = H, W
            cur_K = K
            cur_gt_rgb = gt_rgb
            if has_depth:
                cur_gt_depth_flat = gt_depth_flat
                cur_valid_mask = valid_depth_mask
        
        # --- COMPUTE POSE ---
        # Rodrigues formula for rotation delta
        theta = torch.norm(rot_delta_param)
        if theta < 1e-6:
            R_delta = torch.eye(3, device=device)
        else:
            k = rot_delta_param / theta
            K_mat = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=device)
            R_delta = torch.eye(3, device=device) + torch.sin(theta)*K_mat + (1-torch.cos(theta))*(K_mat @ K_mat)
            
        w2c_curr = torch.eye(4, device=device)
        w2c_curr[:3, :3] = R_delta @ R_init
        w2c_curr[:3, 3] = trans_param
        
        # --- RENDER (Using YOUR original rasterizer) ---
        # This calls your rasterize_soft internally
        render_out = model(w2c_curr, cur_K, cur_H, cur_W, mode='soft')
        
        render_rgb = render_out[..., :3]
        render_depth = render_out[..., 3]
        
        # --- LOSS ---
        loss_rgb = torch.abs(render_rgb - cur_gt_rgb).mean()
        
        loss_depth = 0.0
        if has_depth:
            # Only compare valid depth pixels
            diff = torch.abs(render_depth.reshape(-1)[cur_valid_mask] - cur_gt_depth_flat[cur_valid_mask])
            loss_depth = diff.mean() if diff.numel() > 0 else 0.0
            
        total_loss = loss_rgb + 0.1 * loss_depth
        
        total_loss.backward()
        optimizer_track.step()
        
    # --- 4. FINALIZE ---
    with torch.no_grad():
        theta = torch.norm(rot_delta_param)
        if theta < 1e-6:
            R_delta = torch.eye(3, device=device)
        else:
            k = rot_delta_param / theta
            K_mat = torch.tensor([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], device=device)
            R_delta = torch.eye(3, device=device) + torch.sin(theta)*K_mat + (1-torch.cos(theta))*(K_mat @ K_mat)
            
        w2c_final = torch.eye(4, device=device)
        w2c_final[:3, :3] = R_delta @ R_init
        w2c_final[:3, 3] = trans_param
        
        c2w_final = torch.inverse(w2c_final)
        
    return c2w_final.cpu().numpy()

def optimize_map_window(model, keyframes, K, H, W, iters=10):
    ''' Optimizes Map (Means, Colors, Opacities) using a sliding window of frames '''
    if len(keyframes) == 0: return
    
    if len(keyframes) == 0: return
    
    # Setup Persistent Optimizer
    model.training_setup()
    optimizer = model.optimizer

    
    for _ in range(iters):
        # Randomly sample one frame from window (Stochastic Gradient Descent)
        idx = np.random.randint(0, len(keyframes))
        frame = keyframes[idx]
        
        c2w = frame['c2w']
        gt_rgb = frame['rgb'].to(model.means.device) 
        gt_depth = frame.get('depth')
        if gt_depth is not None: gt_depth = gt_depth.to(model.means.device)
        
        # Prepare View
        if isinstance(c2w, np.ndarray):
             c2w = torch.tensor(c2w, dtype=torch.float32, device=model.means.device)
        w2c = torch.inverse(c2w)
        
        # Render
        out = model(w2c, K, H, W, mode='soft')
        render_rgb = out[..., :3]
        
        loss = torch.abs(render_rgb - gt_rgb).mean()
        
        if gt_depth is not None:
            render_depth = out[..., 3]
            valid_mask = (gt_depth > 0.1) & (gt_depth < 8.0)
            if valid_mask.any():
                loss_depth = torch.abs(render_depth.reshape(-1)[valid_mask.reshape(-1)] - gt_depth.reshape(-1)[valid_mask.reshape(-1)]).mean()
                loss += 0.1 * loss_depth
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def spawn_gaussians_from_frame(frame_data, K, H, W, c2w_override=None, mode="sensor", subsample=4):
    device = "cuda" # Spawn on cuda usually to save GPU mem, then move
    img = np.array(Image.open(frame_data['rgb_path']).convert("RGB")) / 255.0
    
    if mode == "unidepth":
        depth = np.load(frame_data['unidepth_path'])
    else:
        depth_png = np.array(Image.open(frame_data['sensor_depth_path']))
        depth_png = cv2.medianBlur(depth_png, 5) 
        depth = depth_png.astype(np.float32) / 5000.0
        depth[depth == 0] = -1.0
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    ys, xs = np.indices((H, W))
    ys, xs = ys[::subsample, ::subsample], xs[::subsample, ::subsample]
    z = depth[::subsample, ::subsample]
    img_small = img[::subsample, ::subsample]
    
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    xyz_cam = np.stack([x, y, z], axis=-1)
    
    c2w = c2w_override if c2w_override is not None else frame_data['c2w']
    
    xyz_world = (c2w[:3, :3] @ xyz_cam.reshape(-1, 3).T).T + c2w[:3, 3]
    colors = img_small.reshape(-1, 3)
    
    mask = (z.reshape(-1) > 0.1) & (z.reshape(-1) < 8.0)
    
    # Return on SAME DEVICE as Model is expected to use? 
    # Usually we move it later. For now, return cuda tensors is safe.
    return torch.tensor(xyz_world[mask], dtype=torch.float32),            torch.tensor(colors[mask], dtype=torch.float32)

def get_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    return -10 * torch.log10(mse)

def save_ply(model, filename):
    means = model.means.detach().cpu().numpy()
    colors = torch.sigmoid(model.colors).detach().cpu().numpy()

    
    vertex = np.array([tuple(np.concatenate([means[i], colors[i]*255])) for i in range(len(means))],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filename)
    print(f"Saved {filename}")

def visualize_ply(ply_path, subsample=10):
    print("Loading PLY...")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    x = vertex['x'][::subsample]
    y = vertex['y'][::subsample]
    z = vertex['z'][::subsample]
    r = vertex['red'][::subsample]
    g = vertex['green'][::subsample]
    b = vertex['blue'][::subsample]
    
    colors = np.stack([r, g, b], axis=-1) / 255.0
    
    # Simple outlier filter
    def filter_outliers(arr):
        q5, q95 = np.percentile(arr, 5), np.percentile(arr, 95)
        return (arr >= q5) & (arr <= q95)
        
    mask = filter_outliers(x) & filter_outliers(y) & filter_outliers(z)
    x, y, z = x[mask], y[mask], z[mask]
    colors = colors[mask]
    
    print(f"Visualizing {len(x)} points...")
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=z, z=y,
        mode='markers',
        marker=dict(size=2, color=colors, opacity=0.8)
    )])
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()


