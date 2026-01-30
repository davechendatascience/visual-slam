import cv2
import numpy as np

from slam.core.math_utils import pose_inv, pose_compose


class RGBDTracker:

    def __init__(self, K, min_matches=30, debug_dir=None):
        self.K = K
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.debug_dir = debug_dir
        if self.debug_dir:
            import os
            os.makedirs(self.debug_dir, exist_ok=True)
            self.frame_count = 0

    def extract(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, desc = self.orb.detectAndCompute(gray, None)
        return kps, desc

    def track(self, prev_kf, curr_rgb, curr_depth, init_c2w=None):
        if prev_kf is None or prev_kf.descriptors is None:
            return prev_kf.c2w if prev_kf else np.eye(4, dtype=np.float32), 0.0

        kps2, des2 = self.extract(curr_rgb)
        if des2 is None or len(kps2) < self.min_matches:
            return prev_kf.c2w, 0.0

        matches = self.bf.match(prev_kf.descriptors, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        if len(matches) < self.min_matches:
            return prev_kf.c2w, 0.0

        pts_3d = []
        pts_2d = []
        match_indices = []
        for i, m in enumerate(matches):
            u, v = prev_kf.keypoints[m.queryIdx].pt
            z = prev_kf.depth[int(v), int(u)]
            if z <= 0.1 or z > 8.0:
                continue
            x = (u - self.K[0, 2]) * z / self.K[0, 0]
            y = (v - self.K[1, 2]) * z / self.K[1, 1]
            # Transform point from Prev Camera to World to Current Camera
            # Actually PnP needs Points in World (or Prev Frame) and 2D in Current.
            # Here we define 3D points in *Previous Keyframe System*.
            # The result rvec/tvec will be T_curr_prev.
            pts_3d.append([x, y, z])
            u2, v2 = kps2[m.trainIdx].pt
            pts_2d.append([u2, v2])
            match_indices.append(i)

        if len(pts_3d) < self.min_matches:
            return prev_kf.c2w, 0.0

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Handle extrinsic guess
        use_guess = False
        rvec_init = None
        tvec_init = None
        
        if init_c2w is not None:
             # We are solving for T_curr_prev.
             # T_curr_prev = (T_world_curr)^-1 * T_world_prev
             # But init_c2w is T_world_curr_guess
             # So T_curr_prev_guess = inv(init_c2w) @ prev_kf.c2w
             w2c_curr_guess = pose_inv(init_c2w)
             T_curr_prev_guess = w2c_curr_guess @ prev_kf.c2w
             
             rvec_init, _ = cv2.Rodrigues(T_curr_prev_guess[:3, :3])
             tvec_init = T_curr_prev_guess[:3, 3]
             use_guess = True

        # ensure float64 and proper shape
        if rvec_init is not None:
             rvec_init = np.array(rvec_init, dtype=np.float64)
        if tvec_init is not None:
             tvec_init = np.array(tvec_init, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            self.K,
            None,
            rvec=rvec_init,
            tvec=tvec_init,
            useExtrinsicGuess=use_guess,
            iterationsCount=100,
            reprojectionError=2.0,
            flags=cv2.SOLVEPNP_ITERATIVE 
        )
        if not success:
            return prev_kf.c2w, 0.0

        # Motion Sanity Check
        # Check 1: Inlier Count (Robustness)
        num_inliers = len(inliers) if inliers is not None else 0
        
        # Check 2: Velocity/Translation Magnitude (Physics)
        # However, tvec from PnP is T_curr_prev (Position of Prev in Current Cam).
        # Actually PnP with objectPoints (Prev frame) and imagePoints (Current frame) finds T_world_curr.
        # Wait, let's verify my PnP setup again.
        # pts_3d are X,Y,Z in *Previous Keyframe* coordinate system.
        # pts_2d are u,v in *Current Frame*.
        # solvePnP finds T_prev_curr (Transform from 3D Prev points to Current Camera).
        # So X_curr = R * X_prev + t.
        # t is the position of the Previous Origin in Current Camera Frame.
        # The magnitude of t is the distance between Previous Center and Current Center.
        # So norm(tvec) is indeed the distance moved.
        
        t_dist = np.linalg.norm(tvec)
        
        is_sanity_fail = False
        fail_reason = ""
        
        if num_inliers < 15:
            is_sanity_fail = True
            fail_reason = f"Low Inliers ({num_inliers})"
        elif t_dist > 0.2: # 20cm per frame is Huge (6m/s at 30fps)
            is_sanity_fail = True
            fail_reason = f"Large Jump ({t_dist:.3f}m)"

        if is_sanity_fail:
             print(f"[Sanity] Tracking Failed: {fail_reason}. Fallback to Motion Model.")
             
             if self.debug_dir:
                 # Force save failure case
                 import os
                 img_fail = cv2.drawMatches(
                    prev_kf.rgb, prev_kf.keypoints,
                    curr_rgb, kps2,
                    [matches[match_indices[i]] for i in inliers.flatten()] if inliers is not None else [],
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                 )
                 cv2.imwrite(os.path.join(self.debug_dir, f"failure_{self.frame_count:05d}.jpg"), img_fail)
                 self.frame_count += 1
             
             # Fallback
             if init_c2w is not None:
                 # Use the motion model guess
                 return init_c2w, 0.0
             else:
                 # Assume no motion
                 return prev_kf.c2w, 0.0

        R, _ = cv2.Rodrigues(rvec)
        T_curr_prev = np.eye(4, dtype=np.float32)
        T_curr_prev[:3, :3] = R
        T_curr_prev[:3, 3] = tvec.squeeze()
        
        # C_curr = C_prev * (T_curr_prev)^-1
        # Wait, PnP finds Transform from 3D points (Prev space) to Camera (Curr)
        # So X_curr = R * X_prev + t  => T_curr_prev
        # We want world pose: T_world_curr = T_world_prev * (T_curr_prev)^-1
        
        T_prev_curr = pose_inv(T_curr_prev)
        c2w_curr = pose_compose(prev_kf.c2w, T_prev_curr)

        inlier_ratio = float(len(inliers)) / float(len(pts_3d)) if inliers is not None else 0.0
        
        if self.debug_dir and self.frame_count % 5 == 0:
            import os
            # Visualize matches
            img_matches = cv2.drawMatches(
                prev_kf.rgb, prev_kf.keypoints,
                curr_rgb, kps2,
                [matches[match_indices[i]] for i in inliers.flatten()] if inliers is not None else [],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(os.path.join(self.debug_dir, f"match_{self.frame_count:05d}.jpg"), img_matches)
            
        if self.debug_dir:
            self.frame_count += 1
            
        return c2w_curr, inlier_ratio


def should_make_keyframe(curr_c2w, last_kf_c2w, inlier_ratio, min_trans=0.05, min_rot_deg=15.0):
    if curr_c2w is not None and getattr(curr_c2w, "ndim", 0) == 3:
        curr_c2w = curr_c2w[0]
    if last_kf_c2w is not None and getattr(last_kf_c2w, "ndim", 0) == 3:
        last_kf_c2w = last_kf_c2w[0]
    if last_kf_c2w is None:
        return True
    delta_t = np.linalg.norm(curr_c2w[:3, 3] - last_kf_c2w[:3, 3])
    R_delta = curr_c2w[:3, :3] @ last_kf_c2w[:3, :3].T
    # Handle numerical instability
    val = (np.trace(R_delta) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    angle = np.degrees(np.arccos(val))
    if delta_t > min_trans or angle > min_rot_deg:
        return True
    if inlier_ratio < 0.3:
        return True
    return False

