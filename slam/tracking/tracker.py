import cv2
import numpy as np

from slam.core.math_utils import pose_inv, pose_compose


class RGBDTracker:
    def __init__(self, K, min_matches=30):
        self.K = K
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches

    def extract(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, desc = self.orb.detectAndCompute(gray, None)
        return kps, desc

    def track(self, prev_kf, curr_rgb, curr_depth):
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
        for m in matches:
            u, v = prev_kf.keypoints[m.queryIdx].pt
            z = prev_kf.depth[int(v), int(u)]
            if z <= 0.1 or z > 8.0:
                continue
            x = (u - self.K[0, 2]) * z / self.K[0, 0]
            y = (v - self.K[1, 2]) * z / self.K[1, 1]
            pts_3d.append([x, y, z])
            u2, v2 = kps2[m.trainIdx].pt
            pts_2d.append([u2, v2])

        if len(pts_3d) < self.min_matches:
            return prev_kf.c2w, 0.0

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            self.K,
            None,
            iterationsCount=100,
            reprojectionError=2.0,
        )
        if not success:
            return prev_kf.c2w, 0.0

        R, _ = cv2.Rodrigues(rvec)
        T_curr_prev = np.eye(4, dtype=np.float32)
        T_curr_prev[:3, :3] = R
        T_curr_prev[:3, 3] = tvec.squeeze()

        w2c_prev = pose_inv(prev_kf.c2w)
        w2c_curr = pose_compose(T_curr_prev, w2c_prev)
        c2w_curr = pose_inv(w2c_curr)

        inlier_ratio = float(len(inliers)) / float(len(pts_3d)) if inliers is not None else 0.0
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
    angle = np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
    if delta_t > min_trans or angle > min_rot_deg:
        return True
    if inlier_ratio < 0.3:
        return True
    return False

