import os
import sys
import h5py
import json
import numpy as np
import imageio
from pathlib import Path

# Add LIBERO to path
LIBERO_REPO = Path("/home/dhruv/Trajectory_Augmentation/src/third_party/LIBERO")
if str(LIBERO_REPO) not in sys.path:
    sys.path.insert(0, str(LIBERO_REPO))

import libero.libero.envs
from robosuite import make

def main():
    print("Generating full trajectory reversibility video...")
    
    demo_path = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets/libero_goal/put_the_wine_bottle_on_the_rack_demo.hdf5"
    
    # Load metadata and state 30
    with h5py.File(demo_path, 'r') as f:
        env_args = json.loads(f['data'].attrs['env_args'])
        demo_key = "demo_0"
        
        states = f[f"data/{demo_key}/states"][:]
        actions = f[f"data/{demo_key}/actions"][:]
        
    # Construct BDDL path
    task_name = Path(demo_path).stem.replace("_demo", "")
    dataset_type = demo_path.split("/")[-2]
    bddl_path = str(LIBERO_REPO / "libero" / "libero" / "bddl_files" / dataset_type / f"{task_name}.bddl")
    
    # Create env with offscreen renderer
    env = make(
        env_name="Libero_Tabletop_Manipulation",
        bddl_file_name=bddl_path,
        robots=env_args["env_kwargs"].get("robots", ["Panda"]),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        control_freq=env_args["env_kwargs"].get("control_freq", 20),
        horizon=1000,
    )
    
    # Reset to initial state
    env.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    
    frames = []
    
    def render_frame():
        obs = env._get_observations()
        img = obs["agentview_image"]
        # Robosuite returns upside down images sometimes, flip it
        img = np.flipud(img)
        frames.append(img)
    
    avg_mag = 0.678342
    
    num_steps = len(actions)
    
    for t in range(num_steps):
        print(f"Processing step {t}/{num_steps}")
        
        # Reset state to ground truth to prevent error compounding over 60 steps
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        
        if t < 60:
            # 1. Render original state s_t
            render_frame()
            
            # Extract original gripper state (e.g. -1.0 for open)
            orig_gripper = actions[t][6]
            
            # Generate random noise action
            random_dir = np.random.randn(6)
            random_dir = random_dir / np.linalg.norm(random_dir)
            noise_action_6d = random_dir * avg_mag
            
            # 7D action (gripper = orig_gripper to not change gripper)
            noise_action_7d = np.append(noise_action_6d, orig_gripper)
            
            # Pad to 8D for environment
            noise_action = np.concatenate([noise_action_7d[:6], [noise_action_7d[6], noise_action_7d[6]]])
            
            # 2. Forward pass (apply noise)
            env.step(noise_action)
            render_frame()  # Render s_t_new
            
            # 3. Inverse pass (apply -noise)
            inverse_action_6d = -noise_action_6d
            inverse_action_7d = np.append(inverse_action_6d, orig_gripper)
            inverse_action = np.concatenate([inverse_action_7d[:6], [inverse_action_7d[6], inverse_action_7d[6]]])
            env.step(inverse_action)
            render_frame()  # Render recovered state
            
            # 4. Apply original action to advance
            orig_action = actions[t]
            orig_action_padded = np.concatenate([orig_action[:6], [orig_action[6], orig_action[6]]])
            env.step(orig_action_padded)
            
        else:
            # For t >= 60, just render state and apply action
            render_frame()
            orig_action = actions[t]
            orig_action_padded = np.concatenate([orig_action[:6], [orig_action[6], orig_action[6]]])
            env.step(orig_action_padded)
            
    # Render the final state
    render_frame()
    
    # Check success
    success = env._check_success()
    print(f"\n{'='*40}")
    print(f"TRAJECTORY SUCCESS: {success}")
    print(f"{'='*40}\n")
        
    # Save video
    output_path = "/home/dhruv/Trajectory_Augmentation/src/report_mds/video/full_trajectory_reversibility.mp4"
    imageio.mimsave(output_path, frames, fps=2)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
