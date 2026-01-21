# Visual SLAM Implementation Guide (Step by Step)

This guide is the source of truth for building the 3-tier SLAM pipeline in this
repo. It is organized as an ordered checklist so you can implement and validate
each step without guessing what comes next.

---

## 0) Prerequisites

- GPU recommended (CUDA) for DROID-SLAM and any dense tracking.
- Python env with `requirements.txt` installed.
- TUM RGB-D sequence (fr3_office) configured in `configs/tum_fr3_office.yaml`.

---

## 1) Baseline 3-Tier Skeleton

Goal: confirm the pipeline runs with CPU ORB-PnP tracking and CPU mapping.

Checklist:
1. `run_slam.py` loads config and dataset.
2. Tracking thread processes frames and outputs poses.
3. Mapping thread ingests keyframes and adds Gaussians.
4. Loop-closure thread runs asynchronously.
5. Outputs: `outputs/trajectory.txt` and `outputs/map.ply`.

Validation:
- Run `python run_slam.py --config configs/tum_fr3_office.yaml`.
- Viewer optional; check map and trajectory outputs.

---

## 2) DROID-SLAM Integration (Chosen Tracking)

Goal: replace CPU tracking with DROID-SLAM pose inference.

Checklist:
1. Clone DROID-SLAM: `https://github.com/princeton-vl/DROID-SLAM`
2. Set `tracking.mode: droid` in config.
3. Set `tracking.droid.repo_path` to the clone path.
4. Verify `DroidSLAMAdapter` imports and runs.

Validation:
- Run and confirm tracking uses DROID-SLAM (add a log if needed).
- Compare ATE vs ORB-PnP baseline.

---

## 3) Mapping: 3D Gaussian Map (Current)

Goal: map is represented as 3D Gaussians with RGB colors.

Current behavior:
- Backproject depth to 3D points.
- Add as Gaussians (mean + color + fixed scale + opacity).
- Prune by opacity.

Validation:
- Check `outputs/map.ply`.
- Visualize with `python view_ply.py --ply outputs/map.ply`.

---

## 4) Loop Closure (Async)

Goal: global drift correction without blocking tracking.

Current behavior:
- ORB descriptors for loop detection (CPU).
- PnP geometry check.
- Apply global correction in tracking loop.

Validation:
- Check logs for loop closure events.
- Confirm trajectory shifts are applied globally.

---

## 5) Performance Targets

Initial targets:
- Tracking at ≥ 15 FPS on a GPU.
- Mapping at 1–5 Hz.
- Stable loop closure without blocking tracking.

Knobs:
- `runtime.log_every_n` to reduce logging overhead.
- `visualization.live` to disable live viewer.

---

## 6) Accuracy & Stability Checks

Checklist:
1. No GT pose used for reconstruction unless explicitly enabled.
2. ATE/RPE evaluation saved in outputs.
3. Map density stays bounded (pruning).

---

## 7) Research Notes (Why DROID-SLAM)

- DROID-SLAM is a strong GPU tracking front-end with high accuracy.
- It is widely used as a baseline for dense tracking.
- It can be paired with our 3DGS mapper for a fast + stable pipeline.

Reference: https://github.com/princeton-vl/DROID-SLAM

