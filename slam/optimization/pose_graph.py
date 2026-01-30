import numpy as np
import cv2
from scipy.optimize import least_squares
from slam.core.math_utils import pose_inv, pose_compose

class PoseGraphOptimizer:
    def __init__(self):
        self.loop_constraints = [] # List of (kf_idx_A, kf_idx_B, T_B_from_A)

    def add_loop_constraint(self, idx1, idx2, relative_pose):
        """
        Add a loop closure constraint.
        idx1, idx2: Indices in the keyframes list.
        relative_pose: T_2_from_1 (Transformation from 1 to 2) 4x4
        """
        self.loop_constraints.append((idx1, idx2, relative_pose))

    def optimize(self, keyframes, fixed_indices=[0], max_iter=20):
        """
        Perform Pose Graph Optimization on linked keyframes.
        
        Args:
            keyframes: List of Keyframe objects (will be modified in-place).
            fixed_indices: Indices of keyframes to fix (anchor).
        """
        n_poses = len(keyframes)
        if n_poses < 2:
            return

        # 1. State Vector: [rvec_0, tvec_0, rvec_1, tvec_1, ...]
        # We optimize w2c poses because it's slightly more intuitive for projection,
        # but here we deal with relative poses. State can be c2w or w2c.
        # Let's use c2w (World Pose of Camera) as state.
        
        initial_params = np.zeros((n_poses, 6))
        for i, kf in enumerate(keyframes):
            # Store c2w as rvec, tvec
            # Note: Rodrigues on rotation matrix
            rvec, _ = cv2.Rodrigues(kf.c2w[:3, :3])
            tvec = kf.c2w[:3, 3]
            initial_params[i, :3] = rvec.squeeze()
            initial_params[i, 3:] = tvec

        x0 = initial_params.flatten()

        # 2. Build Odometry Constraints (Sequential)
        # We assume keyframes are temporally ordered.
        # Calculate Delta_ij (relative motion) from INITIAL poses and treat as measurement.
        # T_j = T_i * Delta_ij  => Delta_ij = T_i^-1 * T_j
        # We want to preserve these relative motions unless loops pull them apart.
        odometry_constraints = []
        for i in range(n_poses - 1):
            T_i = keyframes[i].c2w
            T_j = keyframes[i+1].c2w
            # Measured relative motion (from initial trajectory)
            Delta_ij_meas = pose_inv(T_i) @ T_j
            odometry_constraints.append((i, i+1, Delta_ij_meas))

        # 3. Error Function
        def error_func(x):
            params = x.reshape((n_poses, 6))
            residuals = []
            
            # Helper to reconstruct Matrix from params
            def get_matrix(idx):
                r = params[idx, :3]
                t = params[idx, 3:]
                R, _ = cv2.Rodrigues(r)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                return T

            # A. Odometry Residuals
            # Error = Log( Delta_meas^-1 * (T_i^-1 * T_j) )
            # Or simply: Diff between (T_i^-1 * T_j) and Delta_meas
            # We use Log map approximation: Translation diff + Rotation vector diff
            
            w_odo = 1.0 # Weight for odometry
            
            for (i, j, Delta_meas) in odometry_constraints:
                Ti = get_matrix(i)
                Tj = get_matrix(j)
                
                # Estimated Relative: Delta_est = Ti^-1 * Tj
                Delta_est = pose_inv(Ti) @ Tj
                
                # Error: Delta_err = Delta_meas^-1 * Delta_est  (Should be Identity)
                Delta_err = pose_inv(Delta_meas) @ Delta_est
                
                # Log map (approx)
                # Translation part
                res_t = Delta_err[:3, 3]
                # Rotation part (Rodrigues magnitude)
                r_vec, _ = cv2.Rodrigues(Delta_err[:3, :3])
                res_r = r_vec.squeeze()
                
                residuals.extend(res_t * w_odo)
                residuals.extend(res_r * w_odo * 2.0) # Rotation weight

            # B. Loop Residuals
            w_loop = 5.0 # Higher weight for loops? Or robust kernel?
            # Actually, robust loss (Huber/Cauchy) in least_squares handles outliers.
            
            for (i, j, Delta_loop) in self.loop_constraints:
                # Delta_loop is T_j_from_i? 
                # Our add_loop_constraint said T_B_from_A.
                # So T_j = T_i * Delta_loop
                
                if i >= n_poses or j >= n_poses:
                    continue # ID mismatch if optimization window shifted?
                
                Ti = get_matrix(i)
                Tj = get_matrix(j)
                
                Delta_est = pose_inv(Ti) @ Tj
                Delta_err = pose_inv(Delta_loop) @ Delta_est
                
                res_t = Delta_err[:3, 3]
                r_vec, _ = cv2.Rodrigues(Delta_err[:3, :3])
                res_r = r_vec.squeeze()
                
                residuals.extend(res_t * w_loop)
                residuals.extend(res_r * w_loop * 2.0)

            # C. Anchor Residual (Soft constraint to keep T0 fixed)
            # Alternatively, remove T0 from optimization vars?
            # scipy least_squares doesn't support removing vars comfortably with structure.
            # We can add a strong prior on T0.
            if len(fixed_indices) > 0:
                w_fix = 1000.0
                for idx in fixed_indices:
                    T_curr = get_matrix(idx)
                    T_target = keyframes[idx].c2w # Keep it where it was
                    
                    T_err = pose_inv(T_target) @ T_curr
                    res_t = T_err[:3, 3]
                    r_vec, _ = cv2.Rodrigues(T_err[:3, :3])
                    res_r = r_vec.squeeze()
                    
                    residuals.extend(res_t * w_fix)
                    residuals.extend(res_r * w_fix)

            return np.array(residuals)

        # 4. Solve
        # verbose=0 for speed
        res = least_squares(error_func, x0, verbose=0, method='trf', ftol=1e-3, loss='huber', max_nfev=max_iter)
        
        # 5. Update Keyframes
        final_params = res.x.reshape((n_poses, 6))
        for i in range(n_poses):
            r = final_params[i, :3]
            t = final_params[i, 3:]
            R, _ = cv2.Rodrigues(r)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            keyframes[i].c2w = T

        return res.cost
