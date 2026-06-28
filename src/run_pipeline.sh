#!/bin/bash
set -e

echo "=========================================================="
echo "  LIBERO Trajectory Augmentation & Conversion Pipeline    "
echo "=========================================================="
echo ""

# Configuration
CONDA_ENV="traj_aug"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_CMD="python"

# Ensure we're in the right directory
cd "$BASE_DIR"

echo "[1/3] Generating Augmented Datasets via Robosuite MuJoCo..."
echo "      This will perturb the expert trajectories for libero_goal, libero_spatial, and libero_object."
echo "      (Note: This requires EGL rendering setup and may take hours depending on CPU/GPU)"
echo "----------------------------------------------------------"
# Set environment variables for headless rendering
export MUJOCO_GL="egl"
export EGL_DEVICE_ID="0"

# Run the augmentation script
$PYTHON_CMD src/augmentation/run_all_augmentations.py
echo ">>> Augmentation complete!"
echo ""

echo "[2/3] Converting Augmented HDF5 files to LeRobot v3.0 format..."
echo "      This will map 110/79D states to 8D OpenCV-compliant actions and compress videos via SVT-AV1."
echo "----------------------------------------------------------"

cd "$BASE_DIR/src/libero_to_lerobot/libero2lerobot"
# Run the conversion script
bash convert.sh
echo ">>> Conversion complete!"
echo ""

echo "[3/3] Pipeline Finished Successfully!"
echo "----------------------------------------------------------"
echo "The fully formatted LeRobot dataset is now available at:"
echo "$BASE_DIR/data/lerobot_format/"
echo ""
echo "You can load it natively using:"
echo "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset"
echo "dataset = LeRobotDataset('JDhruvR/libero_trajectory_augmented/libero_goal')"
