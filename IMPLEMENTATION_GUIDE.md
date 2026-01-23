# Visual SLAM Implementation Guide (Step by Step)

This guide is the source of truth for building the 3-tier SLAM pipeline in this
repo. It is organized as an ordered checklist so you can implement and validate
each step without guessing what comes next.

---

## 0) Prerequisites

- GPU recommended (CUDA) for dense tracking and mapping.
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

## 2) Mapping: 3D Gaussian Map (Current)

Goal: map is represented as 3D Gaussians with RGB colors.

Current behavior:
- Backproject depth to 3D points.
- Add as Gaussians (mean + color + fixed scale + opacity).
- Prune by opacity.

Validation:
- Check `outputs/map.ply`.
- Visualize with `python view_ply.py --ply outputs/map.ply`.

---

## 3) Loop Closure (Async)

Goal: global drift correction without blocking tracking.

Current behavior:
- ORB descriptors for loop detection (CPU).
- PnP geometry check.
- Apply global correction in tracking loop.

Validation:
- Check logs for loop closure events.
- Confirm trajectory shifts are applied globally.

---

## 4) Performance Targets

Initial targets:
- Tracking at ≥ 15 FPS on a GPU.
- Mapping at 1–5 Hz.
- Stable loop closure without blocking tracking.

Knobs:
- `runtime.log_every_n` to reduce logging overhead.
- `visualization.live` to disable live viewer.

---

## 5) Accuracy & Stability Checks

Checklist:
1. No GT pose used for reconstruction unless explicitly enabled.
2. ATE/RPE evaluation saved in outputs.
3. Map density stays bounded (pruning).

---
