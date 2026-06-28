import os
import sys
import h5py
import json
import numpy as np
import imageio
from pathlib import Path
import glob

# Add LIBERO to path dynamically
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LIBERO_REPO = BASE_DIR / "src" / "third_party" / "LIBERO"
if str(LIBERO_REPO) not in sys.path:
    sys.path.insert(0, str(LIBERO_REPO))

import libero.libero.envs
from robosuite import make
import robosuite.utils.transform_utils as T

def render_trajectory(env, initial_state, actions, output_path):
    print(f"Rendering trajectory of length {len(actions)} to {output_path}...")
    frames = []
    
    # Set initial state
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    
    # Render first frame
    obs = env._get_observations()
    img = obs["agentview_image"]
    frames.append(np.flip(img, axis=(0, 1)).astype(np.uint8))
    
    for t, action in enumerate(actions):
        # Handle 7D vs 8D action padding if needed
        action_padded = np.concatenate([action[:6], [action[6], action[6]]]) if len(action) == 7 else action
        env.step(action_padded)
        
        obs = env._get_observations()
        img = obs["agentview_image"]
        
        # Apply 180-degree flip to fix mujoco camera orientation
        img_flipped = np.flip(img, axis=(0, 1))
        frames.append(img_flipped.astype(np.uint8))
        
        if t % 50 == 0:
            print(f"  Rendered {t}/{len(actions)} frames")
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=20, format='FFMPEG', macro_block_size=None)
    print(f"Saved {output_path}\n")

def main():
    print("Generating 256x256 demo videos for presentation...")
    
    suites = ["libero_goal", "libero_spatial", "libero_object"]
    base_data_dir = BASE_DIR / "data"
    
    # Find one successful task for each suite
    for suite in suites:
        print(f"\n{'='*50}\nProcessing {suite}\n{'='*50}")
        
        orig_dir = base_data_dir / "LIBERO-datasets" / suite
        aug_dir = base_data_dir / "LIBERO-datasets-augmented" / suite
        
        if not orig_dir.exists() or not aug_dir.exists():
            print(f"Directories for {suite} missing, skipping...")
            continue
            
        # Get first task that exists in both
        orig_files = list(orig_dir.glob("*.hdf5"))
        if not orig_files:
            print(f"No original files for {suite}")
            continue
            
        task_name = orig_files[0].stem.replace("_demo", "")
        
        orig_path = str(orig_dir / f"{task_name}_demo.hdf5")
        aug_path = str(aug_dir / f"{task_name}_demo.hdf5")
        
        if not os.path.exists(aug_path):
            print(f"Augmented file not found for {task_name}, skipping...")
            continue
            
        print(f"Selected task: {task_name}")
        
        try:
            # Load metadata
            with h5py.File(orig_path, 'r') as f:
                env_args = json.loads(f['data'].attrs['env_args'])
                orig_initial_state = f["data/demo_0/states"][0]
                orig_actions = f["data/demo_0/actions"][:]
                
            with h5py.File(aug_path, 'r') as f:
                # Get the first two augmented trajectories
                demo_keys = list(f['data'].keys())
                if len(demo_keys) < 2:
                    print(f"Not enough augmented trajectories in {aug_path}")
                    continue
                
                aug1_initial_state = f[f"data/{demo_keys[0]}/states"][0]
                aug1_actions = f[f"data/{demo_keys[0]}/actions"][:]
                
                aug2_initial_state = f[f"data/{demo_keys[1]}/states"][0]
                aug2_actions = f[f"data/{demo_keys[1]}/actions"][:]
                
            # Construct BDDL path
            bddl_path = str(LIBERO_REPO / "libero" / "libero" / "bddl_files" / suite / f"{task_name}.bddl")
            
            # Create env with offscreen renderer at 256x256
            env = make(
                env_name=env_args.get("env_name", "Libero_Tabletop_Manipulation"),
                bddl_file_name=bddl_path,
                robots=env_args.get("env_kwargs", {}).get("robots", ["Panda"]),
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names="agentview",
                camera_heights=256,
                camera_widths=256,
                control_freq=env_args.get("env_kwargs", {}).get("control_freq", 20),
                horizon=1000,
            )
            
            out_dir = BASE_DIR / "videos" / suite
            
            # Render Original
            env.reset()
            render_trajectory(env, orig_initial_state, orig_actions, str(out_dir / "original_trajectory.mp4"))
            
            # Render First Perturbed
            env.reset()
            render_trajectory(env, aug1_initial_state, aug1_actions, str(out_dir / "perturbed_1.mp4"))
            
            # Render Second Perturbed
            env.reset()
            render_trajectory(env, aug2_initial_state, aug2_actions, str(out_dir / "perturbed_2.mp4"))
            
            # Close env to free EGL resources
            env.close()
            
        except Exception as e:
            print(f"Failed processing {suite}: {e}")

if __name__ == "__main__":
    main()
