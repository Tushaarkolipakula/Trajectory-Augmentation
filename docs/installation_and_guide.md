# Installation & Usage Guide

This document details the exact steps required to replicate our entire environment, run the trajectory augmentation pipeline, and generate the final Hugging Face LeRobot dataset.

## System Requirements
- OS: Linux (Ubuntu 20.04+ recommended)
- GPU: NVIDIA GPU with CUDA support (required for MuJoCo EGL rendering and LeRobot video compression).
- Storage: ~50GB free space for generating the full augmented datasets across all suites.
- Memory: 32GB+ RAM.

---

## 1. Environment Setup

We recommend using `conda` or `mamba` to create the isolated environment. The exact dependencies from our working setup have been frozen in `requirements.txt`.

### Clone the Repository
```bash
git clone https://github.com/JDhruvR/Trajectory_Augmentation.git
cd Trajectory_Augmentation
```

### Create the Conda Environment
```bash
conda create -n traj_aug python=3.10 -y
conda activate traj_aug
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Acquiring the Raw LIBERO Datasets

Before augmenting the trajectories, you must download the raw expert demonstrations. Our bundled LIBERO codebase provides an automated script to fetch these directly from Hugging Face into your `data/` folder.

```bash
python src/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
    --download-dir data/LIBERO-datasets \
    --datasets all \
    --use-huggingface
```

---

## 3. Running the End-to-End Pipeline

We have created a unified script (`run_pipeline.sh`) that automates the entire process:
1. It simulates the MuJoCo environments for `libero_goal`, `libero_spatial`, and `libero_object`.
2. It extracts the original trajectories and injects 6D inverse-kinematics noise right before the critical grasp action.
3. It generates `2` new augmented HDF5 trajectories for every `1` original expert demonstration.
4. It passes the resulting augmented HDF5 files to the datatrove-based LeRobot converter, mapping the LIBERO state vector to an 8D OpenCV-compliant schema.
5. It compresses the image streams using SVT-AV1 and outputs the final parquet format.

### Execution
Run the pipeline directly from the root of the repository:

```bash
conda activate traj_aug
bash src/run_pipeline.sh
```

**Note on Execution Time:**
Depending on your CPU core count and GPU capabilities, the full suite augmentation may take several hours. The script utilizes multiprocessing internally to max out the CPU. 

It runs completely headless using `MUJOCO_GL=egl`.

---

## 4. Directory Outputs

After the script finishes successfully, you will find the results located at:

```
Trajectory_Augmentation/
└── data/
    ├── LIBERO-datasets-augmented/    # Raw generated HDF5 files (Robosuite/LIBERO native)
    └── lerobot_format/               # The final Hugging Face v3.0 formatted dataset
        └── libero_trajectory_augmented/
            ├── libero_goal
            ├── libero_object
            └── libero_spatial
```

## 5. Usage in Training

You can immediately load the output into the Hugging Face ecosystem using the `LeRobotDataset` API natively:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Path to the local generated folder
dataset = LeRobotDataset("JDhruvR/libero_trajectory_augmented/libero_goal", local_files_only=True)

# You can now pass this directly into the SmolVLA or Act configs for training!
```
