# Technical Stack & OSS Libraries

This document outlines the core technologies and Open Source Software (OSS) libraries that power our trajectory augmentation pipeline.

## Core Technical Stack

- **Language:** Python 3.10
- **Physics Engine:** MuJoCo (via `mujoco` Python bindings)
- **Rendering:** EGL (headless hardware-accelerated rendering)
- **Data Format:** HDF5 (for raw state/action tracking) and Parquet/SVT-AV1 (for Hugging Face LeRobot datasets)

## Open Source Libraries Used

Our project heavily relies on the following OSS projects to handle environment simulation, robot control, and dataset packaging.

1. **[Robosuite](https://github.com/ARISE-Initiative/robosuite)**
   - **Role:** Provides the core physics simulation environments for robotic manipulation. We use it to load and roll out the base environments before applying our noise injections.
   
2. **[LIBERO Benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)**
   - **Role:** The source of our base expert demonstrations. We use their provided 130 task definitions across the spatial, object, and goal suites to generate our augmented variants.

3. **[Hugging Face LeRobot](https://github.com/huggingface/lerobot)**
   - **Role:** The target format ecosystem. We utilize LeRobot's v3.0 dataset tools to package our augmented HDF5 data into highly compressed, standardized Vision-Language-Action (VLA) training formats.

4. **[Datatrove](https://github.com/huggingface/datatrove)**
   - **Role:** The underlying parallel processing engine used by the LeRobot v3.0 converter to quickly chunk and compress massive amounts of video data.

5. **[h5py](https://www.h5py.org/)**
   - **Role:** Used for reading, writing, and parsing the complex, nested hierarchical data formats exported by the MuJoCo engine during trajectory rollouts.

6. **[NumPy](https://numpy.org/)**
   - **Role:** Handles the underlying mathematical operations required for inverse kinematics calculations, matrix transformations, and coordinate frame conversions.
