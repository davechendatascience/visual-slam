import cv2
import numpy as np

from slam.core.math_utils import pose_inv, pose_compose


class LoopDetector:
    def __init__(self, K, min_matches=40, score_thresh=0.3):
        self.K = K
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.score_thresh = score_thresh
        self.database = []

    def add_keyframe(self, keyframe):
        self.database.append(keyframe)

    def detect_loop(self, curr_kf):
        if len(self.database) < 5:
            return None
        best = None
        best_score = 0.0
        for kf in self.database[:-5]:
            if kf.descriptors is None or curr_kf.descriptors is None:
                continue
            matches = self.bf.match(kf.descriptors, curr_kf.descriptors)
            if len(matches) < self.min_matches:
                continue
            score = len(matches) / max(len(kf.descriptors), 1)
            if score > best_score:
                best_score = score
                best = (kf, matches)

        if best is None or best_score < self.score_thresh:
            return None

        kf_ref, matches = best
        pts_3d = []
        pts_2d = []
        for m in matches:
            u, v = kf_ref.keypoints[m.queryIdx].pt
            z = kf_ref.depth[int(v), int(u)]
            if z <= 0.1 or z > 8.0:
                continue
            x = (u - self.K[0, 2]) * z / self.K[0, 0]
            y = (v - self.K[1, 2]) * z / self.K[1, 1]
            pts_3d.append([x, y, z])
            u2, v2 = curr_kf.keypoints[m.trainIdx].pt
            pts_2d.append([u2, v2])

        if len(pts_3d) < self.min_matches:
            return None

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            self.K,
            None,
            iterationsCount=200,
            reprojectionError=2.0,
        )
        if not success or inliers is None or len(inliers) < self.min_matches:
            return None

        R, _ = cv2.Rodrigues(rvec)
        T_curr_ref = np.eye(4, dtype=np.float32)
        T_curr_ref[:3, :3] = R
        T_curr_ref[:3, 3] = tvec.squeeze()

        w2c_ref = pose_inv(kf_ref.c2w)
        w2c_curr = pose_compose(T_curr_ref, w2c_ref)
        c2w_curr_aligned = pose_inv(w2c_curr)

        return {
            "ref_id": kf_ref.keyframe_id,
            "curr_id": curr_kf.keyframe_id,
            "c2w_aligned": c2w_curr_aligned,
        }

