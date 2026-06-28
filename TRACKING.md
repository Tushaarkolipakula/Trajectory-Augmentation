# Trajectory Augmentation Tracking & Setup Guide

This document tracks the current state of the repository, environment setup instructions, and the correct commands for validating inverse actions.

## 1. Environment Setup

The repository relies on the `qwen-vla` conda environment and the LIBERO benchmark dependencies.

### Conda Environment
Activate the required conda environment:
```bash
conda activate qwen-vla
```

### Installing LIBERO
The `third_party/LIBERO` package must be installed in editable mode:
```bash
cd third_party/LIBERO
pip install -e .
```

## 2. Running Inverse Action Validation

We validate the inverse actions by applying the reverse action physically in a MuJoCo simulation to ensure we arrive back at the original state (`s1`). 

**Crucial Configuration:** MuJoCo headless rendering often hangs over SSH. You *must* run these commands with hardware-accelerated EGL explicitly enabled.

### Run Single Demonstration Test
```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 conda run -n qwen-vla python src/test_inverse_kinematics.py
```

### Run Batch Validation (across multiple demos)
```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 conda run -n qwen-vla python src/test_inverse_kinematics.py
```

## 3. Dataset Compatibility Notes & Learnings

The repository was migrated from an older project (`SGVLA-Export`) that operated on the `libero_object` dataset. The current codebase operates on the `libero_goal` dataset.

### Key Technical Differences:
1. **State Dimensionality:**
   - `libero_object` (old): 110-dimensional state (`PickPlaceBread` environment).
   - `libero_goal` (current): 79-dimensional state (`Libero_Tabletop_Manipulation` environment).
2. **Environment Initialization:**
   - Previous versions hardcoded `PickPlaceBread`, which caused massive tracking failures (34% error) when applied to the 79-dimensional `libero_goal` dataset because of dimension layout mismatches.
   - The current `scripts/` now dynamically parse the exact `.bddl` file from the demonstration's HDF5 metadata and dynamically instantiate the correct `Libero_Tabletop_Manipulation` environment.
3. **State Recovery Mechanism:**
   - Previous scripts manually sliced positions and velocities `[:nq]` and `[nq:nq+nv]`. In a 79-dimensional layout, the first index `[0]` represents `time`. Manual slicing shifted all positions by 1 index, causing catastrophic physics failures.
   - We now strictly use `sim.set_state_from_flattened(target_state)`, which is perfectly robust to any underlying dimension layout changes.

### Current Status
**Validation is PASSING**. The dynamic loading methodology achieves a reconstruction error of exactly **~1.87%**, perfectly aligning with the "properly right" baseline from the original `SGVLA-Export` reference.

## 4. Dataset Migration (LeRobot v3.0)

We have successfully migrated the custom augmented LIBERO datasets to the official HuggingFace LeRobot v3.0 format (`.parquet` and AV1-encoded `.mp4` video chunks).

### Conversions Performed
- `libero_goal`
- `libero_object`
- `libero_spatial`

All converted formats reside in `/home/dhruv/Trajectory_Augmentation/data/lerobot_format/`.

### Conversion Pipeline
The `libero_to_lerobot/` directory contains the community-supported `generic_converter` and `libero2lerobot` tools. The pipeline uses `datatrove` to multiprocess episodes and automatically outputs correct LeRobot Dataset schemas (with dimensions fixed natively to `128x128x3`).

## 5. Repository Structure

- `src/`: Contains all core Python code. This includes the consolidated test scripts for inverse kinematics and the logic to build augmented datasets.
- `src/report_mds/`: Output directory where metrics, statistics, and visual histograms are saved after running the simulator tests.
- `src/third_party/`: Contains the forked LIBERO source code.
- `src/libero_to_lerobot/`: Contains the official datatrove-based converter to output LeRobot v3.0 formatted datasets.
- `SGVLA-Export/`: The untouched reference implementation from the previous project.
