import queue
import threading
import time

import numpy as np

from slam.core.data_types import Frame, Keyframe
from slam.tracking.tracker import RGBDTracker, should_make_keyframe
from slam.tracking.photometric_tracker import PhotometricTracker
from slam.mapping.mapper import GaussianMapper
from slam.loop.loop_closure import LoopDetector


class SLAMSystem:
    def __init__(self, config, gaussian_map, K):
        self.config = config
        self.K = K
        self.map = gaussian_map
        self.map_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.frame_queue = queue.Queue(maxsize=config["runtime"]["frame_queue_size"])
        self.keyframe_queue = queue.Queue(maxsize=config["runtime"]["keyframe_queue_size"])
        self.loop_queue = queue.Queue(maxsize=config["runtime"]["loop_queue_size"])
        self.correction_queue = queue.Queue()

        self.poses = []
        self.keyframes = []

        self.feature_extractor = RGBDTracker(K)
        tracking_mode = config["tracking"].get("mode", "orb_pnp")
        if tracking_mode == "photometric":
            self.pose_tracker = PhotometricTracker(
                K,
                device=config["runtime"]["device"],
                max_points=int(config["tracking"].get("photometric", {}).get("max_points", 8000)),
                iters=int(config["tracking"].get("photometric", {}).get("iters", 20)),
                lr=float(config["tracking"].get("photometric", {}).get("lr", 1e-2)),
            )
        else:
            self.pose_tracker = self.feature_extractor
        self.mapper = GaussianMapper(
            gaussian_map,
            K,
            subsample=config["mapping"]["subsample"],
        )
        self.loop_detector = LoopDetector(K)

        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.mapping_thread = threading.Thread(target=self._mapping_loop, daemon=True)
        self.loop_thread = threading.Thread(target=self._loop_loop, daemon=True)
        self.frame_count = 0

    def start(self):
        self.tracking_thread.start()
        self.mapping_thread.start()
        self.loop_thread.start()

    def stop(self):
        self.stop_event.set()
        self.tracking_thread.join()
        self.mapping_thread.join()
        self.loop_thread.join()

    def push_frame(self, frame: Frame):
        self.frame_queue.put(frame)

    def _tracking_loop(self):
        prev_kf = None
        last_kf_c2w = None
        keyframe_id = 0
        prev_pose = None
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.time()
            if prev_kf is None:
                kps, desc = self.feature_extractor.extract(frame.rgb)
                prev_kf = Keyframe(
                    keyframe_id=keyframe_id,
                    timestamp=frame.timestamp,
                    rgb=frame.rgb,
                    depth=frame.depth,
                    c2w=frame.c2w,
                    keypoints=kps,
                    descriptors=desc,
                )
                with self.state_lock:
                    self.keyframes.append(prev_kf)
                    self.poses.append((frame.timestamp, frame.c2w))
                self.keyframe_queue.put(prev_kf)
                self.loop_queue.put(prev_kf)
                last_kf_c2w = frame.c2w
                keyframe_id += 1
                continue

            if isinstance(self.pose_tracker, PhotometricTracker):
                c2w, inlier_ratio = self.pose_tracker.track(self.map, prev_kf.c2w, frame.rgb, depth=frame.depth)
            else:
                c2w, inlier_ratio = self.pose_tracker.track(prev_kf, frame.rgb, frame.depth)

            if c2w is not None and getattr(c2w, "ndim", 0) == 3:
                c2w = c2w[0]
            with self.state_lock:
                self.poses.append((frame.timestamp, c2w))

            if should_make_keyframe(c2w, last_kf_c2w, inlier_ratio):
                kps, desc = self.feature_extractor.extract(frame.rgb)
                kf = Keyframe(
                    keyframe_id=keyframe_id,
                    timestamp=frame.timestamp,
                    rgb=frame.rgb,
                    depth=frame.depth,
                    c2w=c2w,
                    keypoints=kps,
                    descriptors=desc,
                )
                with self.state_lock:
                    self.keyframes.append(kf)
                self.keyframe_queue.put(kf)
                self.loop_queue.put(kf)
                last_kf_c2w = c2w
                prev_kf = kf
                keyframe_id += 1
            t1 = time.time()

            global_update_ms = 0.0
            while True:
                try:
                    delta = self.correction_queue.get_nowait()
                except queue.Empty:
                    break
                update_start = time.time()
                with self.state_lock:
                    for i, (ts, pose) in enumerate(self.poses):
                        self.poses[i] = (ts, delta @ pose)
                    for i, kfi in enumerate(self.keyframes):
                        self.keyframes[i].c2w = delta @ kfi.c2w
                with self.map_lock:
                    self.map.apply_global_transform(delta)
                global_update_ms += (time.time() - update_start) * 1000.0

            self.frame_count += 1
            log_every = self.config["runtime"].get("log_every_n", 1)
            if self.frame_count % log_every == 0:
                tracking_ms = (t1 - t0) * 1000.0
                total_ms = (time.time() - t0) * 1000.0
                dtrans = 0.0
                drot = 0.0
                if prev_pose is not None:
                    dtrans = float(np.linalg.norm(c2w[:3, 3] - prev_pose[:3, 3]))
                    R_delta = c2w[:3, :3] @ prev_pose[:3, :3].T
                    drot = float(
                        np.degrees(
                            np.arccos(
                                np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
                            )
                        )
                    )
                print(
                    f"[TRACK] frame={self.frame_count} "
                    f"tracking_ms={tracking_ms:.2f} "
                    f"global_update_ms={global_update_ms:.2f} "
                    f"total_ms={total_ms:.2f} "
                    f"dtrans={dtrans:.4f} "
                    f"drot={drot:.2f}"
                )
            prev_pose = c2w.copy()

    def _mapping_loop(self):
        while not self.stop_event.is_set():
            try:
                kf = self.keyframe_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.map_lock:
                self.mapper.add_keyframe(kf)
                if self.map.means.shape[0] > self.config["mapping"]["max_gaussians"]:
                    self.map.prune(self.config["mapping"]["min_opacity"])

    def _loop_loop(self):
        while not self.stop_event.is_set():
            try:
                kf = self.loop_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            loop = self.loop_detector.detect_loop(kf)
            self.loop_detector.add_keyframe(kf)
            if loop is None:
                continue

            # Apply a simple global correction: align current keyframe to loop ref
            corrected = loop["c2w_aligned"]
            delta = corrected @ np.linalg.inv(kf.c2w)
            self.correction_queue.put(delta)

    def get_poses(self):
        return list(self.poses)
