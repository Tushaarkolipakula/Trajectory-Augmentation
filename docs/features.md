# Salient Features

This document outlines the core capabilities and standout features of our trajectory augmentation framework.

## 1. Zero Human Teleoperation
The pipeline generates thousands of novel, robust trajectories without requiring a single minute of human teleoperation or manual virtual reality input. It autonomously mines variability from existing expert demonstrations.

## 2. True Physics-Based Perturbation
Unlike standard image augmentations (cropping, color jittering, flipping) that merely change the visual appearance of the dataset, our pipeline physically alters the spatial configuration of the robotic arm within the MuJoCo simulation. 

## 3. Explicit Recovery Training
Because the arm is physically displaced mid-trajectory, the generated dataset forces downstream Vision-Language-Action (VLA) models to learn explicit recovery behaviors. This prevents the "compounding error" problem where a model slightly strays from the golden path and fails because it has never seen a non-ideal state.

## 4. Universal Compatibility
The output of this pipeline is natively formatted for Hugging Face LeRobot (`v3.0`). The datasets can be plugged directly into modern architectures like `SmolVLA`, `Qwen-VLA`, or `ACT` via the `LeRobotDataset` API without any custom data loaders or parsers.

## 5. Scalable Architecture
The entire framework is heavily parallelized and runs completely headless using EGL hardware acceleration. It is designed to scale horizontally across multiple CPU cores and GPUs, allowing it to process massive dataset suites in hours rather than days.

## 6. Comprehensive Validation Suite
The project includes a robust test suite (`src/tests/`) that computationally verifies inverse kinematics resolution, dataset integrity, and image orientation matrices, ensuring that the augmented data is mathematically sound before it ever reaches a training cluster.
