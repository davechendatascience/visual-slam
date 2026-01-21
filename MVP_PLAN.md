# MVP Plan: Real-Time 3D Gaussian Splatting SLAM (3-Tier)

Goal: validate that a 3-tier architecture (Tracking / Mapping / Loop Closure)
can run a real-time 3D Gaussian Splatting SLAM MVP with a TUM RGB-D sequence.

This plan assumes a fresh reimplementation of `slam_core` with modular,
pluggable components and clear thread boundaries. The MVP prioritizes
stable tracking + incremental Gaussian map updates; loop closure is a
minimal async add-on.

---

## MVP Scope

In-scope:
- Real-time tracking with a Gaussian map as the reference.
- Incremental Gaussian map updates using a sliding window of keyframes.
- Minimal loop-closure detection and global pose correction.
- TUM RGB-D dataset (fr3_office) for initial testing.

Out-of-scope (later milestones):
- IMU fusion, multi-camera, relocalization, dense semantics.
- Full DBoW3/NetVLAD integration.
- Long-term map maintenance beyond a sliding window.

---

## Architecture: 3 Tiers

### 1) Tracking Thread (30-60 Hz)
Purpose: estimate current camera pose assuming map is static.

MVP behavior:
- Input: current RGB (and optionally depth), last pose, current Gaussian map.
- Output: pose estimate + keyframe decision.
- Loss: photometric (rendered vs observed RGB), optional depth loss.
- Fallback: if photometric fails, use RGB-D PnP (sparse features).

Keyframe decision rules:
- Translation > 5cm or rotation > 15deg vs last keyframe.
- Tracking quality drops (inlier ratio, residuals, or photometric loss).
- Fixed time gap (e.g., every 0.5-1.0s if motion is low).

### 2) Mapping Thread (1-5 Hz)
Purpose: update map quality incrementally, sliding-window optimization.

MVP behavior:
- Ingest keyframes from tracking.
- Add Gaussians from keyframe depth (sensor or predicted).
- Optimize only local window (e.g., last 10-20 keyframes).
- Prune low-opacity Gaussians to keep map size manageable.

### 3) Loop Closure Thread (Async)
Purpose: detect revisits and apply global correction without blocking tracking.

MVP behavior:
- Maintain a small image database of keyframes (ORB descriptors).
- Detect loop candidate via descriptor similarity.
- Verify with geometry (essential matrix / PnP + inlier threshold).
- If loop is confirmed, run a lightweight pose-graph optimization.
- Apply corrections to keyframe poses and map (global pose update).

---

## MVP Modules (Suggested Layout)

Root-level modules:
- `slam/`
  - `tracking/` (pose estimation, keyframe decision)
  - `mapping/` (Gaussian map update + optimization)
  - `loop/` (place recognition + pose graph)
  - `data/` (frames, keyframes, map state, queues)
  - `render/` (Gaussian rasterizer abstraction)
  - `datasets/` (TUM RGB-D loader)
  - `utils/` (metrics, logging, IO)
- `configs/`
- `run_slam.py`

Pluggability:
- Tracking: "photometric alignment" or "PnP fallback".
- Mapping: "Gaussian update" strategy is swappable.
- Loop: "ORB BoW" can be replaced later by NetVLAD.

---

## MVP Implementation Steps

### Step 1: Dataset Loader (TUM RGB-D)
Deliverables:
- Parse `rgb.txt`, `depth.txt`, `groundtruth.txt`.
- Timestamp association with tolerance (e.g., 0.02s).
- Frame iterator yielding RGB, depth, pose GT (for eval).

### Step 2: Gaussian Map Representation
Deliverables:
- Parameters: means, scales, rotations, opacities, colors.
- CPU/GPU storage strategy (CPU for inactive, GPU for active window).
- Basic prune / merge functions.

### Step 3: Differentiable Renderer
Deliverables:
- Rasterizer abstraction: CUDA (if available) or pure PyTorch fallback.
- Render RGB (and depth if needed).
- Output: `render(rgb, depth)` for loss computation.

### Step 4: Tracking Thread MVP
Deliverables:
- Pose update via photometric alignment (optimize pose only).
- Coarse-to-fine pyramids to stabilize fast motion.
- Keyframe logic + tracking quality scores.

### Step 5: Mapping Thread MVP
Deliverables:
- Spawn Gaussians from keyframe depth.
- Optimize Gaussians in a sliding window.
- Prune low-opacity Gaussians.

### Step 6: Loop Closure MVP
Deliverables:
- Keyframe image database (ORB descriptors).
- Similarity search + geometric verification.
- Pose graph optimization + global correction.

### Step 7: Evaluation + Output
Deliverables:
- Save trajectory to TUM format.
- Compute ATE/RPE vs ground truth.
- Save map to PLY.

---

## Real-Time Feasibility Notes

Tracking thread:
- Single pose optimization on GPU is feasible at 30Hz if the renderer is fast.
- Use a strict iteration cap (e.g., 10-30 per frame).

Mapping thread:
- Windowed optimization at 1-5 Hz is feasible if window size is bounded.
- Keyframe gating prevents GPU overload.

Loop closure:
- Use async CPU steps; only occasional pose-graph updates.

---

## MVP Success Criteria

1) Tracking remains stable on TUM fr3_office for 1-2 minutes.
2) Map updates do not exceed GPU memory budget (e.g., < 3-4 GB).
3) End-to-end FPS ≥ 20 with GPU, ≥ 10 with CPU fallback.
4) ATE within a reasonable range for a minimal setup.

---

## Next Steps After MVP

- Improve photometric loss (masking, robust loss).
- Add loop closure via DBoW3 or NetVLAD.
- Extend map lifecycle: merge Gaussians, long-term compression.
- Add relocalization + dynamic scene filtering.


