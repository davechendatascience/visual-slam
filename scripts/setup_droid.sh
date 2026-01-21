#!/usr/bin/env bash
sudo apt-get update
sudo apt-get install -y libeigen3-dev

set -euo pipefail

REPO_PATH="${1:-/home/itrib40351/Documents/GitHub/others/DROID-SLAM}"
VENV_PATH="${2:-/home/itrib40351/Documents/GitHub/my_repos/visual-slam/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "${REPO_PATH}" ]; then
  echo "Cloning DROID-SLAM into ${REPO_PATH} ..."
  git clone https://github.com/princeton-vl/DROID-SLAM "${REPO_PATH}"
fi

echo "Updating DROID-SLAM submodules ..."
(cd "${REPO_PATH}" && git submodule update --init --recursive)

if [ ! -d "${VENV_PATH}" ]; then
  echo "Creating venv at ${VENV_PATH} ..."
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python binary not found: ${PYTHON_BIN}"
    echo "Set PYTHON_BIN=python3.11 (recommended) and rerun."
    exit 1
  fi
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

echo "Installing project requirements ..."
pip install --upgrade pip
pip install -r /home/itrib40351/Documents/GitHub/my_repos/visual-slam/requirements.txt


# third party dependencies
# install third-party modules (this will take a while)

python - <<'PY'
import sys
try:
    import torch
    print(f"python: {sys.version}")
    print(f"torch version: {torch.__version__}")
    print(f"torch path: {torch.__file__}")
except Exception as exc:
    raise SystemExit(
        "Torch is not importable in this venv. "
        "Use PYTHON_BIN=python3.11 to create a compatible venv."
    ) from exc
PY

echo "Installing DROID-SLAM requirements ..."
if [ -f "${REPO_PATH}/requirements.txt" ]; then
  # open3d wheels are often unavailable for newer Python versions
  grep -v -i "^open3d" "${REPO_PATH}/requirements.txt" > /tmp/droid_requirements.txt
  pip install -r /tmp/droid_requirements.txt
fi

echo "Installing DROID-SLAM (editable) ..."
cd "${REPO_PATH}"
PIP_NO_BUILD_ISOLATION=1 pip install -e thirdparty/lietorch --no-build-isolation
PIP_NO_BUILD_ISOLATION=1 pip install -e thirdparty/pytorch_scatter --no-build-isolation
export PIP_NO_BUILD_ISOLATION=1
export EIGEN_INCLUDE_DIR=/usr/include/eigen3
export CPLUS_INCLUDE_PATH=/usr/include/eigen3:${CPLUS_INCLUDE_PATH:-}
export C_INCLUDE_PATH=/usr/include/eigen3:${C_INCLUDE_PATH:-}
pip install -e . --no-build-isolation

echo "Downloading DROID-SLAM weights if script exists ..."
if [ -f "${REPO_PATH}/tools/download_models.sh" ]; then
  (cd "${REPO_PATH}/tools" && bash download_models.sh)
else
  echo "No download_models.sh found. Check DROID-SLAM README for weights."
fi

echo "DROID-SLAM setup complete."

