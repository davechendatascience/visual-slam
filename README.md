# Visual SLAM with 3D Gaussian Splatting

A 3-tier SLAM system (Tracking, Mapping, Loop Closure) using 3D Gaussian Splatting for dense mapping. This project is validated on valid TUM RGB-D sequences.

## 🚀 Features

- **3-Tier Architecture**: 
  - **Tracking**: Real-time pose estimation.
  - **Mapping**: Dense mapping using 3D Gaussian Splatting.
  - **Loop Closure**: Asynchronous global correction.
- **TUM RGB-D Dataset Support**: Automatic download and loading of TUM sequences.
- **Visualization**: Live viewer for the Gaussian map and trajectory.

## 📋 Prerequisites

- **Python**: 3.8+
- **GPU**: CUDA-enabled GPU (Highly Recommended for real-time performance)

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/my_repos/visual-slam.git
    cd visual-slam
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### Running the SLAM System

To run the full SLAM pipeline on the TUM fr3_office sequence:

```bash
python run_slam.py --config configs/tum_fr3_office.yaml
```

**Note**: The script will automatically download and extract the TUM dataset if it's not detected in the configured path.

### Visualization

To visualize the generated dense 3D map (PLY file):

```bash
python view_ply.py --ply outputs/map.ply --normalize
```

## 📂 Project Structure

- `slam/`: Core SLAM logic and submodules.
  - `tracking/`: Pose estimation modules.
  - `mapping/`: Gaussian splatting and map management.
  - `loop/`: Loop closure and pose graph optimization.
- `configs/`: Configuration files (YAML).
- `outputs/`: Default directory for saved trajectories and maps.
- `scripts/`: Helper scripts.
- `IMPLEMENTATION_GUIDE.md`: Detailed step-by-step implementation guide.
- `MVP_PLAN.md`: Minimal Viable Product conceptual plan.

## 📄 License

[MIT](LICENSE)
