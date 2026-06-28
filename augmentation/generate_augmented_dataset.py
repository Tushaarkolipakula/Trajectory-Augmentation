"""
Core multiprocessed pipeline for generating augmented datasets.

This script parses the expert HDF5 demonstrations and dynamically 
spawns headless MuJoCo environments across available CPU cores to 
inject structured physical noise, forcing out-of-distribution 
recovery behavior.
"""
import os
import sys
import glob
import h5py
import json
import numpy as np
import multiprocessing as mp
from pathlib import Path
import traceback
import copy
import robosuite.utils.transform_utils as T

LIBERO_REPO = Path(__file__).resolve().parent.parent / "third_party" / "LIBERO"
if str(LIBERO_REPO) not in sys.path:
    sys.path.insert(0, str(LIBERO_REPO))

import libero.libero.envs
from robosuite import make

def process_task(task_file, output_dir, num_augmentations=5):
    try:
        task_name = Path(task_file).stem.replace("_demo", "")
        dataset_type = Path(task_file).parent.name
        bddl_path = str(LIBERO_REPO / "libero" / "libero" / "bddl_files" / dataset_type / f"{task_name}.bddl")
        
        out_file = Path(output_dir) / Path(task_file).name
        print(f"Processing {task_name} -> {out_file}")
        
        with h5py.File(task_file, 'r') as f_in:
            env_args = json.loads(f_in['data'].attrs['env_args'])
            demo_keys = list(f_in['data'].keys())
            
            env_name = env_args.get("env_name", "Libero_Tabletop_Manipulation")
            
            env_kwargs = env_args.get("env_kwargs", {})
            env_kwargs["has_renderer"] = False
            env_kwargs["has_offscreen_renderer"] = True
            env_kwargs["use_camera_obs"] = True
            if "bddl_file_name" in env_kwargs:
                env_kwargs.pop("bddl_file_name")
            if "env_name" in env_kwargs:
                env_kwargs.pop("env_name")
                
            env = make(
                env_name=env_name,
                bddl_file_name=bddl_path,
                **env_kwargs
            )
            
            avg_mag = 0.678342
            success_count = 0
            
            with h5py.File(out_file, 'w') as f_out:
                data_grp = f_out.create_group('data')
                data_grp.attrs['env_args'] = json.dumps(env_args)
                
                new_demo_idx = 0
                
                for i, demo_key in enumerate(demo_keys):
                    if i % 5 == 0:
                        print(f"Processing demo {i}/{len(demo_keys)}...")
                        
                    orig_states = f_in[f'data/{demo_key}/states'][:]
                    orig_actions = f_in[f'data/{demo_key}/actions'][:]
                    
                    gripper_actions = orig_actions[:, 6]
                    closed_indices = np.where(gripper_actions > 0)[0]
                    if len(closed_indices) > 0:
                        t_grasp = closed_indices[0]
                    else:
                        t_grasp = len(orig_actions)
                        
                    orig_obs_grp = f_in[f'data/{demo_key}/obs']
                    orig_rewards = f_in[f'data/{demo_key}/rewards'][:]
                    orig_dones = f_in[f'data/{demo_key}/dones'][:]
                    orig_robot_states = f_in[f'data/{demo_key}/robot_states'][:]
                    
                    for _ in range(num_augmentations):
                        t_thresh = t_grasp - 5
                        if t_thresh <= 5:
                            t = 5
                        else:
                            t = np.random.randint(5, t_thresh + 1)
                            
                        env.reset()
                        env.sim.set_state_from_flattened(orig_states[t])
                        env.sim.forward()
                        
                        orig_gripper = orig_actions[t][6]
                        random_dir = np.random.randn(6)
                        random_dir = random_dir / np.linalg.norm(random_dir)
                        noise_6d = random_dir * avg_mag
                        noise_7d = np.append(noise_6d, orig_gripper)
                        noise_8d = np.concatenate([noise_7d[:6], [noise_7d[6], noise_7d[6]]])
                        
                        # Apply noise to get the noisy observation
                        obs_dict, reward, done, info = env.step(noise_7d)
                        
                        noisy_state = env.sim.get_state().flatten()
                        noisy_action = noise_7d
                        
                        # Extract the required observation fields from the noisy step
                        # Note: Robosuite camera names use '_image' instead of '_rgb'
                        agentview_img = obs_dict["agentview_image"]
                        eye_in_hand_img = obs_dict["robot0_eye_in_hand_image"]
                        
                        # Apply 180-degree rotation (flip both axes) to make frames right-side up
                        agentview_img = np.flip(agentview_img, axis=(0, 1))
                        eye_in_hand_img = np.flip(eye_in_hand_img, axis=(0, 1))
                        
                        noisy_ee = np.hstack((obs_dict["robot0_eef_pos"], T.quat2axisangle(obs_dict["robot0_eef_quat"])))
                        
                        # We apply the inverse recovery only to advance physics (even though we will directly stitch)
                        inverse_7d = np.append(-noise_6d, orig_gripper)
                        env.step(inverse_7d)
                        
                        # STITCHING LOGIC
                        # 1. Stitch Actions & States
                        new_actions = [noisy_action]
                        new_states = [noisy_state, orig_states[t]]
                        for k in range(t, len(orig_actions)):
                            new_actions.append(orig_actions[k])
                            if k < len(orig_actions) - 1:
                                new_states.append(orig_states[k + 1])
                                
                        # 2. Stitch Metadata (Rewards, Dones, Robot States)
                        new_rewards = np.concatenate([[0.0], orig_rewards[t:]])
                        new_dones = np.concatenate([[0], orig_dones[t:]])
                        
                        # Construct robot_state manually based on LIBERO's implementation
                        # since robosuite raw env doesn't have get_robot_state_vector
                        noisy_robot_state = np.concatenate([
                            obs_dict["robot0_gripper_qpos"],
                            obs_dict["robot0_eef_pos"],
                            obs_dict["robot0_eef_quat"]
                        ])
                        
                        new_robot_states = np.concatenate([np.expand_dims(noisy_robot_state, axis=0), orig_robot_states[t:]], axis=0)
                        
                        # 3. Save to HDF5
                        grp = data_grp.create_group(f'demo_{new_demo_idx}')
                        grp.create_dataset('states', data=np.array(new_states))
                        grp.create_dataset('actions', data=np.array(new_actions))
                        grp.create_dataset('rewards', data=np.array(new_rewards))
                        grp.create_dataset('dones', data=np.array(new_dones))
                        grp.create_dataset('robot_states', data=np.array(new_robot_states))
                        
                        # 4. Stitch Observations
                        obs_grp = grp.create_group('obs')
                        
                        # Apply 180-degree rotation to original frames as well
                        orig_agentview_flipped = np.flip(orig_obs_grp['agentview_rgb'][t:], axis=(1, 2))
                        orig_eye_in_hand_flipped = np.flip(orig_obs_grp['eye_in_hand_rgb'][t:], axis=(1, 2))
                        
                        obs_grp.create_dataset('agentview_rgb', data=np.concatenate([np.expand_dims(agentview_img, axis=0), orig_agentview_flipped], axis=0))
                        obs_grp.create_dataset('eye_in_hand_rgb', data=np.concatenate([np.expand_dims(eye_in_hand_img, axis=0), orig_eye_in_hand_flipped], axis=0))
                        obs_grp.create_dataset('gripper_states', data=np.concatenate([np.expand_dims(obs_dict["robot0_gripper_qpos"], axis=0), orig_obs_grp['gripper_states'][t:]], axis=0))
                        obs_grp.create_dataset('joint_states', data=np.concatenate([np.expand_dims(obs_dict["robot0_joint_pos"], axis=0), orig_obs_grp['joint_states'][t:]], axis=0))
                        obs_grp.create_dataset('ee_states', data=np.concatenate([np.expand_dims(noisy_ee, axis=0), orig_obs_grp['ee_states'][t:]], axis=0))
                        obs_grp.create_dataset('ee_pos', data=np.concatenate([np.expand_dims(noisy_ee[:3], axis=0), orig_obs_grp['ee_pos'][t:]], axis=0))
                        obs_grp.create_dataset('ee_ori', data=np.concatenate([np.expand_dims(noisy_ee[3:], axis=0), orig_obs_grp['ee_ori'][t:]], axis=0))
                        
                        grp.attrs['num_samples'] = len(new_actions)
                        grp.attrs['model_file'] = f_in[f'data/{demo_key}'].attrs.get('model_file', '')
                        grp.attrs['init_state'] = f_in[f'data/{demo_key}'].attrs.get('init_state', new_states[0])
                        
                        new_demo_idx += 1
                        success_count += 1
                            
        print(f"Task {task_name}: Generated {success_count} successful augmented trajectories out of {len(demo_keys) * num_augmentations}")
        return True
    except Exception as e:
        print(f"Error processing {task_file}: {e}")
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    default_base = str(Path(__file__).resolve().parent.parent.parent)
    parser.add_argument("--target_dir", type=str, default=f"{default_base}/data/LIBERO-datasets/libero_goal")
    parser.add_argument("--output_dir", type=str, default=f"{default_base}/data/LIBERO-datasets-augmented/libero_goal")
    parser.add_argument("--num_augmentations", type=int, default=2)
    parser.add_argument("--test_single", action="store_true", help="Run only on a single file for testing")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    hdf5_files = glob.glob(os.path.join(args.target_dir, "*.hdf5"))
    if args.test_single:
        # Just use the wine bottle demo
        hdf5_files = [f for f in hdf5_files if "put_the_wine_bottle_on_the_rack_demo" in f]
        
    print(f"Found {len(hdf5_files)} tasks to process.")
    
    # We use multiprocessing Pool to run multiple environments in parallel
    if len(hdf5_files) > 1:
        pool_args = [(f, args.output_dir, args.num_augmentations) for f in hdf5_files]
        with mp.Pool(processes=min(mp.cpu_count(), len(hdf5_files))) as pool:
            pool.starmap(process_task, pool_args)
    else:
        for f in hdf5_files:
            process_task(f, args.output_dir, args.num_augmentations)

if __name__ == "__main__":
    main()
