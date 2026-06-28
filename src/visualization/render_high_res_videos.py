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

def render_trajectory(env, initial_state, actions, output_path):
    print(f"Rendering trajectory of length {len(actions)} to {output_path}...")
    frames = []
    
    # Set initial state
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    
    # Render first frame
    obs = env._get_observations()
    img = obs["agentview_image"]
    frames.append(np.flipud(img).astype(np.uint8))
    
    for t, action in enumerate(actions):
        action_padded = np.concatenate([action[:6], [action[6], action[6]]]) if len(action) == 7 else action
        env.step(action_padded)
        
        obs = env._get_observations()
        img = obs["agentview_image"]
        frames.append(np.flipud(img).astype(np.uint8))
        
        if t % 20 == 0:
            print(f"  Rendered {t}/{len(actions)} frames")
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=2, format='FFMPEG', macro_block_size=None)
    print(f"Saved {output_path}\n")

def main():
    print("Generating high-resolution videos for original and augmented trajectories...")
    
    suite = "libero_spatial"
    task_name = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
    
    orig_path = f"/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets/{suite}/{task_name}_demo.hdf5"
    aug_path = f"/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets-augmented/{suite}/{task_name}_demo.hdf5"
    
    # Load metadata
    with h5py.File(orig_path, 'r') as f:
        env_args = json.loads(f['data'].attrs['env_args'])
        orig_initial_state = f["data/demo_0/states"][0]
        orig_actions = f["data/demo_0/actions"][:]
        
    with h5py.File(aug_path, 'r') as f:
        aug1_initial_state = f["data/demo_0/states"][0]
        aug1_actions = f["data/demo_0/actions"][:]
        
        aug2_initial_state = f["data/demo_1/states"][0]
        aug2_actions = f["data/demo_1/actions"][:]
        
    # Construct BDDL path
    bddl_path = str(LIBERO_REPO / "libero" / "libero" / "bddl_files" / suite / f"{task_name}.bddl")
    
    # Create env with offscreen renderer
    env = make(
        env_name=env_args["env_name"],
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
    
    env.reset()
    
    # Render Original
    out_dir = "/home/dhruv/Trajectory_Augmentation/src/report_mds/video"
    render_trajectory(env, orig_initial_state, orig_actions, f"{out_dir}/original_trajectory.mp4")
    
    # Render First Perturbed
    render_trajectory(env, aug1_initial_state, aug1_actions, f"{out_dir}/first_perturbed.mp4")
    
    # Render Second Perturbed
    render_trajectory(env, aug2_initial_state, aug2_actions, f"{out_dir}/second_perturbed.mp4")

if __name__ == "__main__":
    main()
