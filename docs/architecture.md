# Technical Architecture

Our solution is built on a two-phase data processing architecture: **Physical Augmentation** and **Format Conversion**. The architecture is designed to handle thousands of robotic trajectories entirely in software using headless rendering.

## 1. Physical Augmentation Pipeline

The core of our solution operates at the physics-engine level rather than the image-processing level. 

1. **Environment Initialization:** We instantiate a `robosuite` MuJoCo environment matching the specific LIBERO task.
2. **Expert Trajectory Rollout:** We load a known expert demonstration from an HDF5 file and replay it within the engine up to a pre-defined critical timestep (e.g., right before a grasp).
3. **State Perturbation:** We dynamically pause the simulation and inject 6-Dimensional noise (position and rotation) into the robot's end-effector state variables.
4. **Recovery Rollout:** The environment is unpaused. The expert policy, using internal inverse kinematics mapping, must calculate a new path from this perturbed out-of-distribution state to complete the original task goal.
5. **Data Capture:** The entire process (states, actions, visual observations) is captured at 20 frames per second using headless EGL hardware rendering and written to a new HDF5 file.

## 2. Format Conversion Pipeline

Once the augmented HDF5 files are generated, they must be converted into a format suitable for modern AI training ecosystems.

1. **Schema Mapping:** A converter script parses the native LIBERO state schema (which varies between 79 and 110 dimensions depending on the suite) and maps it to a strict 8D OpenCV-compliant schema expected by LeRobot.
2. **Parallel Processing:** Using the `datatrove` engine, the pipeline processes the trajectories in chunks, heavily parallelizing the conversion across available CPU cores.
3. **Video Compression:** Raw image arrays from the simulation are encoded into highly efficient SVT-AV1 chunked MP4 video streams.
4. **Metadata Generation:** Finally, Parquet files containing the tabular state/action data are generated alongside dataset metadata (`info.json` and `stats.json`), completing the `v3.0` Hugging Face LeRobot dataset.
