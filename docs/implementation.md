# Implementation Details

Our software stack implements a robust pipeline for handling physics-based trajectory augmentations. Below are the specific details of our implementation.

## Dynamic Noise Injection
Instead of relying on random dataset cropping or basic image transformations, our pipeline directly intercepts the underlying physics simulation in MuJoCo. By interacting with the `robosuite` environment at runtime, we can dynamically manipulate the 6-dimensional coordinate frame of the robot's end effector.

The noise functions generate bounded spherical coordinates, resulting in random translations and rotations that are strictly constrained within a safe radius around the expert demonstration's original path.

## Inverse Kinematics Resolution
After the state has been perturbed, the robotic agent must physically move from this new Out-Of-Distribution (OOD) position to the target object. This requires translating target coordinates into physical joint motor commands.

We achieve this by exposing the internal `OSC_POSE` (Operational Space Control) controller within the LIBERO environment, allowing the underlying inverse kinematics solvers to recalculate a trajectory dynamically without explicit manual pathfinding.

## Multiprocessing Framework
Because physics simulations are heavily CPU-bound and rendering is GPU-bound, generating thousands of trajectories sequentially is incredibly slow. We implemented a robust `concurrent.futures` multiprocessing framework inside `src/augmentation/generate_augmented_dataset.py`.

This framework intelligently partitions the 130 LIBERO tasks across available CPU cores, launching multiple isolated headless environments simultaneously.

## Hardware-Accelerated Rendering
Standard OpenGL rendering requires an active display server (like X11), which fails on remote training servers. We implemented full headless EGL support by configuring `MUJOCO_GL="egl"`. This allows the pipeline to render the camera observations directly on the GPU without a display server, achieving a steady 20 FPS capture rate entirely in the background.

## Data Schema Compliance
A significant engineering challenge was mapping the raw physics output to the Hugging Face LeRobot format. The raw LIBERO state space fluctuates wildly depending on the suite (from 79 dimensions in `libero_goal` to 110 dimensions in `libero_object`). 

We built a custom parser inside `src/libero_to_lerobot/libero2lerobot/` that dynamically masks out environment-specific states, isolates the strict 8-dimensional robot action space (X, Y, Z, Roll, Pitch, Yaw, Gripper, Pad), and corrects the spatial inversion caused by the MuJoCo rendering pipeline, ensuring 100% compliance with native VLA models.
