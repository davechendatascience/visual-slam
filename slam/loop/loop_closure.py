import cv2
import numpy as np
from slam.core.math_utils import pose_inv, pose_compose

class LoopDetector:
    def __init__(self, K):
        self.K = K
        self.keyframes = []
        # Use simple inefficient list for now. In prod, use KDTree.
        
        # Re-use ORB/matcher logic
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = 50 # Higher threshold for loop closure to be safe
        
        # Temporal Consistency
        self.consistency_count = 0
        self.last_loop_candidate_id = -1
        
    def add_keyframe(self, kf):
        self.keyframes.append(kf)
        
    def detect_loop(self, curr_kf):
        """
        Detects loop closure for curr_kf against past keyframes.
        Returns dict with 'c2w_aligned' if loop found, else None.
        """
        curr_pos = curr_kf.c2w[:3, 3]
        best_candidate = None
        
        # 1. Candidate Search (Spatial + Temporal)
        # We search backwards to find the most recent 'old' frame we are close to
        candidates = []
        for kf in reversed(self.keyframes):
            # Temporal constraint: Skip recent ~50 keyframes (assuming 1 KF/sec ~ 50sec buffer)
            if abs(kf.keyframe_id - curr_kf.keyframe_id) < 50:
                continue
                
            # Spatial constraint: Radius < 1.5m
            dist = np.linalg.norm(kf.c2w[:3, 3] - curr_pos)
            if dist < 1.5:
                candidates.append(kf)
                # Heuristic: limit to max 3 candidates to save compute per frame
                if len(candidates) >= 3:
                     break
        
        if not candidates:
            self.consistency_count = 0
            return None
            
        # 2. Geometric Verification
        loop_found = False
        valid_candidate = None
        c2w_aligned_res = None
        
        for candidate in candidates:
            # Match descriptors
            matches = self.bf.match(candidate.descriptors, curr_kf.descriptors)
            matches = sorted(matches, key=lambda m: m.distance)
            
            if len(matches) < self.min_matches:
                continue
                
            pts_3d = []
            pts_2d = []
            
            for m in matches[:200]: # Limit to top 200 matches
                u_cand, v_cand = candidate.keypoints[m.queryIdx].pt
                z_cand = candidate.depth[int(v_cand), int(u_cand)]
                
                if z_cand <= 0.1 or z_cand > 8.0:
                    continue
                
                # Backproject Candidate Point to World using Candidate's KNOWN/FIXED Pose
                x_c = (u_cand - self.K[0, 2]) * z_cand / self.K[0, 0]
                y_c = (v_cand - self.K[1, 2]) * z_cand / self.K[1, 1]
                z_c = z_cand
                
                # Transform to World: P_w = T_c2w_cand * P_c
                p_c = np.array([x_c, y_c, z_c, 1.0])
                p_w = candidate.c2w @ p_c
                
                pts_3d.append(p_w[:3])
                
                # 2D point in Current Frame
                u_curr, v_curr = curr_kf.keypoints[m.trainIdx].pt
                pts_2d.append([u_curr, v_curr])
                
            if len(pts_3d) < self.min_matches:
                continue
                
            pts_3d = np.array(pts_3d, dtype=np.float32)
            pts_2d = np.array(pts_2d, dtype=np.float32)
            
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                self.K,
                None,
                iterationsCount=100,
                reprojectionError=2.0,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if success and inliers is not None and len(inliers) > self.min_matches:
                 # Geometric Verification Passed
                 R, _ = cv2.Rodrigues(rvec)
                 T_curr_world = np.eye(4, dtype=np.float32)
                 T_curr_world[:3, :3] = R
                 T_curr_world[:3, 3] = tvec.squeeze()
                 
                 w2c_aligned = T_curr_world
                 c2w_aligned = pose_inv(w2c_aligned)
                 
                 loop_found = True
                 valid_candidate = candidate
                 c2w_aligned_res = c2w_aligned
                 break
        
        if loop_found:
             # Temporal Consistency Check
             # Check if this candidate (or close neighbor) was found previously
             if self.last_loop_candidate_id != -1 and abs(self.last_loop_candidate_id - valid_candidate.keyframe_id) < 3:
                 self.consistency_count += 1
             else:
                 self.consistency_count = 1
                 
             self.last_loop_candidate_id = valid_candidate.keyframe_id
             
             if self.consistency_count >= 3:
                 print(f"[LOOP] Loop confirmed (consist={self.consistency_count}) between KF {curr_kf.keyframe_id} and {valid_candidate.keyframe_id}!")
                 self.consistency_count = 0 # Reset after triggering
                 return {'c2w_aligned': c2w_aligned_res}
             else:
                  print(f"[LOOP] Potential loop detected (count={self.consistency_count})... waiting for confirmation.")
        else:
             self.consistency_count = 0
             self.last_loop_candidate_id = -1

        return None
