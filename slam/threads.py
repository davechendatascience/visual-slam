import queue
import threading
import time
import os

import numpy as np

from slam.core.data_types import Frame, Keyframe
from slam.optimization.pose_graph import PoseGraphOptimizer
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

        # Setup debug directory
        debug_dir = os.path.join(config["output"]["dir"], "debug_tracking")
        if os.path.exists(debug_dir):
            import shutil
            shutil.rmtree(debug_dir)
        
        self.feature_extractor = RGBDTracker(K, debug_dir=debug_dir)
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
        self.pgo = PoseGraphOptimizer()

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
        velocity = np.eye(4, dtype=np.float32) # c2w_curr = velocity @ c2w_prev

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.time()
            
            # Motion model guess
            init_c2w = None
            if prev_pose is not None:
                # Constant velocity assumption: T_k = T_{k-1} * (T_{k-1}^-1 * T_{k})_prev
                # Actually velocity is delta: P_curr = Delta * P_prev
                init_c2w = velocity @ prev_pose
            
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
                prev_pose = frame.c2w
                keyframe_id += 1
                continue

            if isinstance(self.pose_tracker, PhotometricTracker):
                c2w, inlier_ratio = self.pose_tracker.track(self.map, prev_kf.c2w, frame.rgb, depth=frame.depth)
            else:
                 # Pass motion guess to Feature Tracker
                c2w, inlier_ratio = self.pose_tracker.track(prev_kf, frame.rgb, frame.depth, init_c2w=init_c2w)

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

            # Check for global loop closure corrections
            global_update_ms = 0.0
            while True: # Use True to ensure all corrections are processed
                try:
                    delta = self.correction_queue.get_nowait()
                except queue.Empty:
                    break
                
                update_start = time.time()
                print(f"[TRACK] Applying global correction delta!")
                
                with self.state_lock:
                    # Update all past poses
                    # self.poses is list of (timestamp, c2w)
                    for i, (ts, pose) in enumerate(self.poses):
                        self.poses[i] = (ts, delta @ pose)
                        
                    # Update all keyframes
                    for kf_obj in self.keyframes:
                        kf_obj.c2w = delta @ kf_obj.c2w
                    
                    # Update Keyframes in the queue
                    for kf_obj in self.keyframe_queue.queue:
                        kf_obj.c2w = delta @ kf_obj.c2w
                    
                    # Update Loop Detector Keyframes
                    for kf_obj in self.loop_detector.keyframes:
                        kf_obj.c2w = delta @ kf_obj.c2w
                        
                    # CRITICAL FIX: Update local tracking state to new coordinate system
                    if prev_pose is not None:
                        prev_pose = delta @ prev_pose
                    
                    # Also update current c2w
                    c2w = delta @ c2w
                    
                    # Force-reset velocity model implicitly by ensuring prev_pose matches c2w
                    # (Logic: Next iteration velocity = c2w_new @ inv(prev_pose_new) ~ Identity if we stopped time)
                    # Actually, c2w is current, prev_pose is previous.
                    # Their relative transform should be preserved by rigid delta.
                    # velocity = (delta @ c2w) @ inv(delta @ prev) = delta @ c2w @ inv(prev) @ inv(delta)
                    # = delta @ velocity_old @ inv(delta)
                    # So velocity IS rotated by delta. This is correct!
                    # The code handles physics correctly.
                
                with self.map_lock:
                    self.map.apply_global_transform(delta)
                global_update_ms += (time.time() - update_start) * 1000.0
                print(f"[TRACK] Global Correction Complete. System aligned to Loop Closure. GTErr metrics may shift.")
                
                with self.map_lock:
                    self.map.apply_global_transform(delta)
                global_update_ms += (time.time() - update_start) * 1000.0

            self.frame_count += 1
            log_every = self.config["runtime"].get("log_every_n", 1)
            
            # Update velocity
            if prev_pose is not None:
                # velocity = current * inv(prev)
                from slam.core.math_utils import pose_inv
                velocity = c2w @ pose_inv(prev_pose)
            
            if self.frame_count % log_every == 0:
                tracking_ms = (t1 - t0) * 1000.0
                total_ms = (time.time() - t0) * 1000.0
                dtrans = 0.0
                drot = 0.0
                
                # Debug Drift if GT is available in frame (we don't store GT in frame object currently but it was in input)
                # But wait, frame input to push_frame was Frame object.
                # Frame object has c2w which is GT init (if enabled) but we overwrote it? no track returns new c2w.
                # Frame object is: Timestamp, RGB, Depth, C2W.
                # If use_gt_init was True, frame.c2w passed to push_frame WAS the GT pose (or identity).
                # So we can compare c2w (estimated) with frame.c2w (initial/GT).
                
                gt_err_t = -1.0
                gt_err_r = -1.0
                if self.config["tracking"]["use_gt_init"]: # Assuming passed frame.c2w IS GT
                     safe_gt = frame.c2w
                     if getattr(safe_gt, "ndim", 0) == 3: safe_gt = safe_gt[0]
                     if np.linalg.norm(safe_gt) > 0.1: # Not identity
                         gt_err_t = np.linalg.norm(c2w[:3,3] - safe_gt[:3,3])
                         # rot diff...
                
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
                    f"ms={tracking_ms:.1f}+{global_update_ms:.1f} "
                    f"dpos={dtrans:.3f} "
                    f"GTErr={gt_err_t:.3f}m"
                )
            prev_pose = c2w.copy()

    def _mapping_loop(self):
        while not self.stop_event.is_set():
            try:
                kf = self.keyframe_queue.get(timeout=0.2)
            except queue.Empty:
                continue
                
            # Perform Bundle Adjustment on persistent buffer
            # We do this BEFORE adding to the dense map so the points are accurate?
            # Or AFTER? 
            # If we update pose, `add_keyframe` (which uses kf.c2w) will use the updated pose if we do it first.
            # But the detailed Bundle Adjustment takes time. We don't want to block the map queue too much.
            # Actually, `GaussianMapper.add_keyframe` splats points.
            # If we optimize AFTER, the splatted points are slightly misplaced.
            # Ideally: BA first, then Map.
            
            # Collect window
            with self.state_lock:
                if len(self.keyframes) >= 5:
                    window = self.keyframes[-20:] # Larger window for PGO is fine!
                    
                    # Run PGO
                    t_pgo_start = time.time()
                    self.pgo.optimize(window, fixed_indices=[0])
                    t_pgo_end = time.time()
                    print(f"[PGO] Optimized {len(window)} frames in {(t_pgo_end - t_pgo_start)*1000:.1f}ms")
            
            t_map_start = time.time()
            with self.map_lock:
                self.mapper.add_keyframe(kf)
                if self.map.means.shape[0] > self.config["mapping"]["max_gaussians"]:
                    self.map.prune(self.config["mapping"]["min_opacity"])
            print(f"[MAP] Added Keyframe {kf.keyframe_id}. Map Update: {(time.time() - t_map_start)*1000:.1f}ms")

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
