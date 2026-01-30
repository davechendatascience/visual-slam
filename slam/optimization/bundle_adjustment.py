import numpy as np
import cv2
from scipy.optimize import least_squares
from slam.core.math_utils import pose_inv, pose_compose

class LocalBundleAdjuster:
    def __init__(self, K):
        self.K = K.copy()
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def optimize(self, keyframes, fixed_indices=[0], max_points=1000):
        """
        Perform geometric Bundle Adjustment on a list of Keyframes.
        Optimizes Poses (c2w) and transient 3D Points.
        
        Args:
            keyframes: List of Keyframe objects (will be modified in-place).
            fixed_indices: Indices of keyframes to hold fixed (default: oldest).
            max_points: Max number of sparse points to optimize (for speed).
        """
        n_poses = len(keyframes)
        if n_poses < 2:
            return

        # 1. Build Transient Graph (Feature Tracks)
        # map_points: dict {point_id: 3d_coord}
        # observations: list of (point_id, kf_idx, uv_measured)
        # feature_ref: dict {kf_idx: {feat_idx: point_id}} tracks existing points
        
        point_id_counter = 0
        feature_ref = {i: {} for i in range(n_poses)}
        observations = [] # (point_id, kf_idx, u, v)
        initial_points = {} # point_id -> np.array([x,y,z])

        # Pairwise matching to build tracks
        # We match k with k-1, k-2...
        for i in range(n_poses):
            for j in range(i + 1, n_poses):
                kf1 = keyframes[i]
                kf2 = keyframes[j]
                
                matches = self.matcher.match(kf1.descriptors, kf2.descriptors)
                matches = sorted(matches, key=lambda m: m.distance)
                
                # Filter matches?
                
                for m in matches[:100]: # Limit matches per pair
                    idx1 = m.queryIdx
                    idx2 = m.trainIdx
                    
                    # Check if idx1 already belongs to a point
                    pid = feature_ref[i].get(idx1)
                    
                    if pid is None:
                        # Check idx2?
                        pid = feature_ref[j].get(idx2)
                        
                    if pid is None:
                        # New Point
                        # Initialize 3D position from kf1 (if depth exists)
                        u1, v1 = kf1.keypoints[idx1].pt
                        z1 = kf1.depth[int(v1), int(u1)]
                        
                        if z1 <= 0.1 or z1 > 10.0:
                            continue # Invalid depth, skip new point creation
                            
                        # Backproject to World
                        # X_c = (u - cx) * z / fx
                        x_c = (u1 - self.cx) * z1 / self.fx
                        y_c = (v1 - self.cy) * z1 / self.fy
                        p_c = np.array([x_c, y_c, z1, 1.0])
                        p_w = kf1.c2w @ p_c
                        
                        pid = point_id_counter
                        point_id_counter += 1
                        initial_points[pid] = p_w[:3]
                        
                    # Link observations
                    if idx1 not in feature_ref[i]:
                        feature_ref[i][idx1] = pid
                        observations.append((pid, i, kf1.keypoints[idx1].pt))
                        
                    if idx2 not in feature_ref[j]:
                        feature_ref[j][idx2] = pid
                        observations.append((pid, j, kf2.keypoints[idx2].pt))
                        
        used_point_ids = sorted(list(initial_points.keys()))
        
        if len(used_point_ids) > max_points:
            np.random.shuffle(used_point_ids)
            used_point_ids = used_point_ids[:max_points]
            
        # Filter observations to match selected points
        used_point_set = set(used_point_ids)
        observations = [o for o in observations if o[0] in used_point_set]
        
        if len(used_point_ids) == 0:
            print("[BA] No valid points found (after filtering). Skipping.")
            return

        print(f"[BA] Optimizing {n_poses} frames and {len(used_point_ids)} points.")

        # 2. Prepare Optimization Parameters
        # Params: [pose_0_r, pose_0_t, ..., point_0_x, point_0_y, point_0_z, ...]
        # Poses: each is 6 params (Rodrigues, Translation). c2w?
        # Usually easier to optimize w2c because Proj(X) = K * (R*X + t). 
        # But our state is c2w. We can convert inside error func.
        # Let's optimize w2c parameters directly, then invert back.
        
        pose_params = np.zeros((n_poses, 6))
        for i, kf in enumerate(keyframes):
            w2c = pose_inv(kf.c2w)
            rvec, _ = cv2.Rodrigues(w2c[:3, :3])
            tvec = w2c[:3, 3]
            pose_params[i, :3] = rvec.squeeze()
            pose_params[i, 3:] = tvec
            
        point_params = np.array([initial_points[pid] for pid in used_point_ids]).flatten()
        
        x0 = np.hstack([pose_params.flatten(), point_params])
        
        # 3. Define Error Function
        # We need mapping from obs index to params
        obs_array = np.array(observations, dtype=object) # (N_obs, 3) -> pid, kf_idx, (u,v)
        
        # Map pid (arbitrary) to index in point_params
        pid_to_idx = {pid: i for i, pid in enumerate(used_point_ids)}
        
        obs_p_indices = np.array([pid_to_idx[o[0]] for o in observations], dtype=int)
        obs_kf_indices = np.array([o[1] for o in observations], dtype=int)
        obs_uv = np.array([o[2] for o in observations])
        
        def rep_err(x):
            # Unpack
            curr_poses = x[:n_poses*6].reshape((n_poses, 6))
            curr_points = x[n_poses*6:].reshape((-1, 3))
            
            # Select required poses/points for obs
            p_angs = curr_poses[obs_kf_indices, :3]
            p_trans = curr_poses[obs_kf_indices, 3:]
            pts = curr_points[obs_p_indices]
            
            # Project
            # 1. World to Cam: P_c = R * P_w + t
            # Vectorized implementation needed for speed?
            # Rodrigues is slow to vectorize manually. 
            # Loop is simpler for clarity, but scipy calls this many times.
            # Let's assume N is small (500 pts). Loop over poses might be faster than per-point?
            # Actually, scipy least_squares handles Jacobian...
            
            # Simple per-observation projection
            residuals = []
            
            # To vectorize: Rotate points.
            # But each point is rotated by a different pose.
            # We can group by pose?
            
            # Unoptimized loop for now (Local BA is small)
            projections = np.zeros_like(obs_uv)
            
            # Cache rotation matrices for the N poses
            Rs = []
            ts = []
            for i in range(n_poses):
                R, _ = cv2.Rodrigues(curr_poses[i, :3])
                Rs.append(R)
                ts.append(curr_poses[i, 3:])
                
            for k in range(len(obs_kf_indices)):
                kf_idx = obs_kf_indices[k]
                pt_idx = obs_p_indices[k]
                
                P_w = curr_points[pt_idx]
                P_c = Rs[kf_idx] @ P_w + ts[kf_idx]
                
                # Project
                # u = fx * x / z + cx
                if P_c[2] < 0.001: # Avoid division by zero/behind camera
                    projections[k] = [0,0] # High error
                else:
                    projections[k, 0] = self.fx * P_c[0] / P_c[2] + self.cx
                    projections[k, 1] = self.fy * P_c[1] / P_c[2] + self.cy
                    
            return (projections - obs_uv).flatten()

        # 4. Solve
        # We assume standard least squares
        # print("[BA] Running solver...")
        res = least_squares(rep_err, x0, verbose=0, x_scale='jac', ftol=1e-2, method='trf', max_nfev=5)
        
        # print(f"[BA] Done. Cost: {res.cost:.4f}, Messages: {res.message}")
        
        # 5. Update Keyframes
        optimized_poses = res.x[:n_poses*6].reshape((n_poses, 6))
        
        for i in range(n_poses):
            if i in fixed_indices:
                continue
                
            rvec = optimized_poses[i, :3]
            tvec = optimized_poses[i, 3:]
            
            R, _ = cv2.Rodrigues(rvec)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = tvec
            
            kf_c2w = pose_inv(w2c)
            
            # Update keyframe
            keyframes[i].c2w = kf_c2w
