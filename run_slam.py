import argparse
import os
import shutil
import tarfile
import time
import urllib.request

import numpy as np

from slam.config import load_config
from slam.datasets.tum_rgbd import load_tum_sequence, read_frame
from slam.core.data_types import Frame
from slam.core.gaussian_map import GaussianMap
from slam.core.math_utils import rot_to_quat
from slam.threads import SLAMSystem
from slam.visualization.live_viewer import LiveMapViewer


def save_trajectory(poses, path):
    with open(path, "w", encoding="utf-8") as f:
        for ts, c2w in poses:
            qx, qy, qz, qw = rot_to_quat(c2w[:3, :3])
            tx, ty, tz = c2w[:3, 3]
            f.write(f"{ts:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")


def dataset_ready(root_dir):
    required = ["rgb.txt", "depth.txt", "groundtruth.txt"]
    return all(os.path.exists(os.path.join(root_dir, f)) for f in required)


def ensure_tum_dataset(root_dir, url):
    if dataset_ready(root_dir):
        return

    parent_dir = os.path.dirname(root_dir)
    os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(root_dir) and os.listdir(root_dir):
        raise RuntimeError(
            f"Dataset folder exists but is incomplete: {root_dir}. "
            "Please clear it or fix the contents."
        )

    archive_path = os.path.join(parent_dir, "tum_dataset.tgz")
    print(f"Downloading TUM dataset to {archive_path} ...")
    urllib.request.urlretrieve(url, archive_path)

    print("Extracting dataset ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        top_dirs = {m.name.split("/")[0] for m in tar.getmembers() if m.name}
        tar.extractall(parent_dir)

    os.remove(archive_path)

    extracted_dir = None
    if len(top_dirs) == 1:
        candidate = os.path.join(parent_dir, list(top_dirs)[0])
        if os.path.isdir(candidate):
            extracted_dir = candidate

    if extracted_dir is None:
        raise RuntimeError("Failed to identify extracted dataset directory.")

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    shutil.move(extracted_dir, root_dir)

    if not dataset_ready(root_dir):
        raise RuntimeError("Dataset extraction failed or missing required files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset"]["root"]
    dataset_url = cfg["dataset"].get(
        "url",
        "https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz",
    )
    ensure_tum_dataset(dataset_root, dataset_url)
    frames, depth_scale = load_tum_sequence(dataset_root, cfg["dataset"]["depth_scale"])

    K = np.array(cfg["camera"]["K"], dtype=np.float32)
    device = cfg["runtime"]["device"]
    gaussian_map = GaussianMap(device=device)

    slam = SLAMSystem(cfg, gaussian_map, K)
    slam.start()

    viewer = None
    if cfg.get("visualization", {}).get("live", False):
        viewer = LiveMapViewer(
            gaussian_map,
            slam.map_lock,
            interval_s=cfg["visualization"].get("interval_s", 1.0),
            max_points=cfg["visualization"].get("max_points", 200000),
        )
        viewer.start()

    target_fps = cfg["runtime"]["target_fps"]
    sleep_dt = 1.0 / max(1e-6, target_fps)

    for frame_meta in frames:
        rgb, depth = read_frame(frame_meta, depth_scale)
        c2w_init = frame_meta["c2w_gt"] if cfg["tracking"]["use_gt_init"] else np.eye(4, dtype=np.float32)
        frame = Frame(
            timestamp=frame_meta["timestamp"],
            rgb=rgb,
            depth=depth,
            c2w=c2w_init,
        )
        slam.push_frame(frame)
        time.sleep(sleep_dt)

    time.sleep(1.0)
    slam.stop()
    if viewer is not None:
        viewer.stop()

    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    traj_path = os.path.join(cfg["output"]["dir"], "trajectory.txt")
    save_trajectory(slam.get_poses(), traj_path)
    map_path = os.path.join(cfg["output"]["dir"], "map.ply")
    gaussian_map.to_ply(map_path)


if __name__ == "__main__":
    main()

