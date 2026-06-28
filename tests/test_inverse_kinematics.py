#!/usr/bin/env python3
"""
Test Inverse Kinematics
Validates inverse action reversibility mathematically via the Robosuite simulator.
Tests multiple demos to generate per-demo error bounds and an aggregate validation report.

Usage:
    python test_inverse_kinematics.py [--demo_start 1] [--demo_end 20] [--num_steps 50]

Output:
    - libero_inverse_action_batch_summary.txt: Aggregate statistics
    - libero_inverse_action_batch_demo_*.txt: Per-demo results
    - libero_inverse_action_batch_consistency.png: Error distribution across demos
    - libero_inverse_action_batch_heatmap.png: Error heatmap (demo × step)
"""

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
import json

import robosuite as suite
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat


# ============================================================================
# CONFIGURATION
# ============================================================================

LIBERO_REPO = Path("/home/dhruv/Trajectory_Augmentation/src/third_party/LIBERO")
DEMO_PATH = Path("/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5")
OUTPUT_DIR = Path("/home/dhruv/Trajectory_Augmentation/src/report_mds")

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"LIBERO repo: {LIBERO_REPO}")
print(f"Demo file: {DEMO_PATH}")
print(f"Output dir: {OUTPUT_DIR}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rot_mat_to_quat(rot_mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
    trace = rot_mat.trace()
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (rot_mat[2, 1] - rot_mat[1, 2]) / S
        y = (rot_mat[0, 2] - rot_mat[2, 0]) / S
        z = (rot_mat[1, 0] - rot_mat[0, 1]) / S
    elif (rot_mat[0, 0] > rot_mat[1, 1]) and (rot_mat[0, 0] > rot_mat[2, 2]):
        S = np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2]) * 2
        w = (rot_mat[2, 1] - rot_mat[1, 2]) / S
        x = 0.25 * S
        y = (rot_mat[0, 1] + rot_mat[1, 0]) / S
        z = (rot_mat[0, 2] + rot_mat[2, 0]) / S
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        S = np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2]) * 2
        w = (rot_mat[0, 2] - rot_mat[2, 0]) / S
        x = (rot_mat[0, 1] + rot_mat[1, 0]) / S
        y = 0.25 * S
        z = (rot_mat[1, 2] + rot_mat[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1]) * 2
        w = (rot_mat[1, 0] - rot_mat[0, 1]) / S
        x = (rot_mat[0, 2] + rot_mat[2, 0]) / S
        y = (rot_mat[1, 2] + rot_mat[2, 1]) / S
        z = 0.25 * S
    
    return np.array([x, y, z, w])


def extract_eef_state(env) -> np.ndarray:
    """Extract 8D EEF state from robosuite environment"""
    robot = env.robots[0]
    sim = env.sim
    
    # End-effector position (3D) - from MuJoCo site xpos
    eef_site_id = robot.eef_site_id
    eef_pos = sim.data.site_xpos[eef_site_id]  # (3,)
    
    # End-effector orientation - convert rotation matrix to axis-angle
    eef_rot_mat = sim.data.site_xmat[eef_site_id].reshape((3, 3))  # 9-element vec -> 3x3 mat
    eef_quat = rot_mat_to_quat(eef_rot_mat)
    eef_ori = quat2axisangle(eef_quat)  # (3,)
    
    # Gripper state (2D) - get from joint positions
    gripper_pos = []
    for joint_name in ['gripper0_finger_joint1', 'gripper0_finger_joint2']:
        joint_id = None
        for i, name in enumerate(sim.model.joint_names):
            if name == joint_name:
                joint_id = i
                break
        if joint_id is not None:
            joint_qpos_idx = sim.model.jnt_qposadr[joint_id]
            gripper_pos.append(sim.data.qpos[joint_qpos_idx])
    
    if len(gripper_pos) < 2:
        gripper_pos = [0.0, 0.0]
    
    gripper_state = np.array(gripper_pos[:2], dtype=np.float64)  # (2,)
    eef_state = np.concatenate([eef_pos, eef_ori, gripper_state])
    
    return eef_state  # (8,)


def reset_to_state(env, target_state: np.ndarray) -> None:
    """Reset environment to a specific MuJoCo state (usually 79D or 110D)"""
    sim = env.sim
    sim.set_state_from_flattened(target_state)
    sim.forward()


def load_hdf5_trajectory(filepath: str, demo_idx: int = 0) -> Dict:
    """Load trajectory from HDF5 demo file"""
    with h5py.File(filepath, 'r') as f:
        demo_key = f"data/demo_{demo_idx}"
        
        states = f[f"{demo_key}/states"][:]
        actions = f[f"{demo_key}/actions"][:]
        
        ee_pos = f[f"{demo_key}/obs/ee_pos"][:]
        ee_ori = f[f"{demo_key}/obs/ee_ori"][:]
        gripper_states = f[f"{demo_key}/obs/gripper_states"][:]
    
    return {
        'states_110d': states,
        'actions': actions,
        'ee_pos': ee_pos,
        'ee_ori': ee_ori,
        'gripper_states': gripper_states,
    }


def create_environment(demo_path: str):
    """Create robosuite environment using metadata from the demo file"""
    import json
    import h5py
    import os
    import sys
    
    if str(LIBERO_REPO) not in sys.path:
        sys.path.insert(0, str(LIBERO_REPO))
        
    import libero.libero.envs  # registers the libero envs

    with h5py.File(demo_path, 'r') as f:
        env_args = json.loads(f['data'].attrs['env_args'])
    
    # Construct BDDL path manually
    task_name = Path(demo_path).stem.replace("_demo", "")
    dataset_type = demo_path.split("/")[-2]  # e.g., libero_goal
    bddl_path = str(LIBERO_REPO / "libero" / "libero" / "bddl_files" / dataset_type / f"{task_name}.bddl")
    
    env = suite.make(
        env_name="Libero_Tabletop_Manipulation",
        bddl_file_name=bddl_path,
        robots=env_args["env_kwargs"].get("robots", ["Panda"]),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=env_args["env_kwargs"].get("control_freq", 20),
        horizon=200,
    )
    return env


# ============================================================================
# BATCH TEST FUNCTION
# ============================================================================

def test_single_demo(env, trajectory_data: Dict, demo_idx: int,
                     max_steps: int = 50) -> Dict:
    """Test inverse action reversibility for a single demo"""
    
    states_110d = trajectory_data['states_110d']
    actions = trajectory_data['actions']
    
    num_steps = min(len(states_110d) - 1, max_steps)
    
    results = {
        's1': [],
        'a1': [],
        's2': [],
        's1_reconstructed': [],
        'error': [],
        'error_pos': [],
        'error_ori': [],
        'error_grip': [],
    }
    
    for step in range(num_steps):
        try:
            # Reset to s1
            s1_110d = states_110d[step]
            reset_to_state(env, s1_110d)
            s1_eef = extract_eef_state(env)
            
            # Apply forward action
            a1 = actions[step]
            # Pad action to 8D (robosuite expects [pos(3), ori_axis_angle(3), gripper_l, gripper_r])
            a1_padded = np.concatenate([a1[:6], [a1[6], a1[6]]])
            env.step(a1_padded)
            s2_eef = extract_eef_state(env)
            
            # Apply inverse action
            inverse_action = a1.copy()
            inverse_action[:6] *= -1
            inverse_action[6] = 0
            # Pad to 8D
            inverse_action_padded = np.concatenate([inverse_action[:6], [inverse_action[6], inverse_action[6]]])
            
            env.step(inverse_action_padded)
            s1_reconstructed_eef = extract_eef_state(env)
            
            # Compute errors
            total_error = np.linalg.norm(s1_eef - s1_reconstructed_eef)
            pos_error = np.linalg.norm(s1_eef[:3] - s1_reconstructed_eef[:3])
            ori_error = np.linalg.norm(s1_eef[3:6] - s1_reconstructed_eef[3:6])
            grip_error = np.linalg.norm(s1_eef[6:8] - s1_reconstructed_eef[6:8])
            
            results['s1'].append(s1_eef)
            results['a1'].append(a1)
            results['s2'].append(s2_eef)
            results['s1_reconstructed'].append(s1_reconstructed_eef)
            results['error'].append(total_error)
            results['error_pos'].append(pos_error)
            results['error_ori'].append(ori_error)
            results['error_grip'].append(grip_error)
        
        except Exception as e:
            print(f"    Step {step}: FAILED - {str(e)[:50]}")
            continue
    
    # Convert lists to numpy arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.array(results[key])
    
    return results


def compute_demo_statistics(results: Dict) -> Dict:
    """Compute statistics for a single demo"""
    
    if len(results['error']) == 0:
        return None
    
    errors = results['error']
    errors_pos = results['error_pos']
    errors_ori = results['error_ori']
    errors_grip = results['error_grip']
    
    s1_states = np.array(results['s1'])
    s1_norms = np.linalg.norm(s1_states, axis=1)
    
    stats = {
        'num_steps': len(errors),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'percentile_95': np.percentile(errors, 95),
        'percentile_99': np.percentile(errors, 99),
        'mean_s1_norm': np.mean(s1_norms),
        'std_s1_norm': np.std(s1_norms),
        'error_as_pct': (np.mean(errors) / np.mean(s1_norms)) * 100,
        'error_pos_mean': np.mean(errors_pos),
        'error_ori_mean': np.mean(errors_ori),
        'error_grip_mean': np.mean(errors_grip),
    }
    
    return stats


# ============================================================================
# BATCH TESTING
# ============================================================================

def test_batch(demo_start: int = 1, demo_end: int = 20, num_steps: int = 50) -> Dict:
    """Test multiple demos and collect statistics"""
    
    print(f"\\n=== Batch Inverse Action Validation ===")
    print(f"Testing demos {demo_start} to {demo_end}...")
    print(f"Steps per demo: {num_steps}\\n")
    
    env = create_environment(str(DEMO_PATH))
    
    batch_results = {
        'demo_indices': [],
        'statistics': [],
        'errors_by_demo': [],  # For heatmap
    }
    
    for demo_idx in range(demo_start, demo_end + 1):
        try:
            print(f"Demo {demo_idx:2d}: ", end='', flush=True)
            
            # Load trajectory
            trajectory_data = load_hdf5_trajectory(str(DEMO_PATH), demo_idx=demo_idx)
            
            # Test
            results = test_single_demo(env, trajectory_data, demo_idx, max_steps=num_steps)
            
            # Compute stats
            stats = compute_demo_statistics(results)
            
            if stats is None:
                print("SKIPPED (no valid steps)")
                continue
            
            batch_results['demo_indices'].append(demo_idx)
            batch_results['statistics'].append(stats)
            batch_results['errors_by_demo'].append(np.array(results['error']))
            
            print(f"mean_error={stats['mean_error']:.6f}m ({stats['error_as_pct']:.2f}%), " +
                  f"max_error={stats['max_error']:.6f}m, steps={stats['num_steps']}")
        
        except Exception as e:
            print(f"FAILED - {str(e)[:60]}")
            continue
    
    return batch_results


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_batch_results(batch_results: Dict, output_dir: Path) -> None:
    """Save batch results and generate visualizations"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Aggregate statistics
    stats_list = batch_results['statistics']
    demo_indices = batch_results['demo_indices']
    
    if len(stats_list) == 0:
        print("ERROR: No valid demos tested!")
        return
    
    # Summary file
    summary_file = output_dir / "libero_inverse_action_batch_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("LIBERO Inverse Action - Batch Validation Summary\\n")
        f.write("=" * 100 + "\\n\\n")
        f.write(f"Number of demos tested: {len(stats_list)}\\n")
        f.write(f"Demo indices: {demo_indices}\\n\\n")
        
        # Aggregate error statistics
        mean_errors = [s['mean_error'] for s in stats_list]
        max_errors = [s['max_error'] for s in stats_list]
        error_pcts = [s['error_as_pct'] for s in stats_list]
        
        f.write("AGGREGATE ERROR STATISTICS (across all demos):\\n")
        f.write("-" * 100 + "\\n")
        f.write(f"Mean error (across demos): {np.mean(mean_errors):.6f} m (std: {np.std(mean_errors):.6f} m)\\n")
        f.write(f"Max error (across demos): {np.max(max_errors):.6f} m (min: {np.min(max_errors):.6f} m)\\n")
        f.write(f"Mean error as % of state norm: {np.mean(error_pcts):.2f}% (std: {np.std(error_pcts):.2f}%)\\n\\n")
        
        # Component errors
        pos_errors = [s['error_pos_mean'] for s in stats_list]
        ori_errors = [s['error_ori_mean'] for s in stats_list]
        grip_errors = [s['error_grip_mean'] for s in stats_list]
        
        f.write("COMPONENT ERRORS (mean across demos):\\n")
        f.write("-" * 100 + "\\n")
        f.write(f"Position: {np.mean(pos_errors):.6f} m (range: {np.min(pos_errors):.6f} - {np.max(pos_errors):.6f})\\n")
        f.write(f"Orientation: {np.mean(ori_errors):.6f} rad (range: {np.min(ori_errors):.6f} - {np.max(ori_errors):.6f})\\n")
        f.write(f"Gripper: {np.mean(grip_errors):.6f} (range: {np.min(grip_errors):.6f} - {np.max(grip_errors):.6f})\\n\\n")
        
        # Per-demo breakdown
        f.write("PER-DEMO BREAKDOWN:\\n")
        f.write("-" * 100 + "\\n")
        f.write("Demo | Mean Err | Std Err | Max Err | Min Err | 95th %ile | Err % | Pos | Ori | Grip\\n")
        f.write("-" * 100 + "\\n")
        
        for demo_idx, stats in zip(demo_indices, stats_list):
            f.write(f"{demo_idx:4d} | {stats['mean_error']:8.6f} | {stats['std_error']:7.6f} | "
                   f"{stats['max_error']:7.6f} | {stats['min_error']:7.6f} | {stats['percentile_95']:9.6f} | "
                   f"{stats['error_as_pct']:6.2f}% | {stats['error_pos_mean']:.4f} | "
                   f"{stats['error_ori_mean']:.4f} | {stats['error_grip_mean']:.4f}\\n")
        
        # Validation verdict
        f.write("\\n" + "=" * 100 + "\\n")
        f.write("VALIDATION VERDICT:\\n")
        f.write("-" * 100 + "\\n")
        
        mean_all = np.mean(mean_errors)
        if mean_all < 0.1:
            verdict = "✓ PASS - Mean error < 0.1 m across all demos. Suitable for trajectory augmentation."
        elif mean_all < 0.15:
            verdict = "⚠ MARGINAL - Mean error 0.1-0.15 m. Use with caution."
        else:
            verdict = "✗ FAIL - Mean error > 0.15 m. Not suitable for augmentation."
        
        f.write(verdict + "\\n")
    
    print(f"✓ Saved summary to: {summary_file}")
    
    # Per-demo detail files
    for demo_idx, stats in zip(demo_indices, stats_list):
        detail_file = output_dir / f"libero_inverse_action_batch_demo_{demo_idx:02d}.txt"
        with open(detail_file, 'w') as f:
            f.write(f"Demo {demo_idx} - Inverse Action Statistics\\n")
            f.write("=" * 80 + "\\n\\n")
            f.write(f"Steps tested: {stats['num_steps']}\\n")
            f.write(f"Mean ‖s1‖: {stats['mean_s1_norm']:.6f} m (std: {stats['std_s1_norm']:.6f})\\n\\n")
            f.write(f"Mean error: {stats['mean_error']:.6f} m\\n")
            f.write(f"Std error: {stats['std_error']:.6f} m\\n")
            f.write(f"Min error: {stats['min_error']:.6f} m\\n")
            f.write(f"Max error: {stats['max_error']:.6f} m\\n")
            f.write(f"Median error: {stats['median_error']:.6f} m\\n")
            f.write(f"95th percentile: {stats['percentile_95']:.6f} m\\n")
            f.write(f"99th percentile: {stats['percentile_99']:.6f} m\\n\\n")
            f.write(f"Error as % of state norm: {stats['error_as_pct']:.2f}%\\n\\n")
            f.write("Component errors:\\n")
            f.write(f"  Position: {stats['error_pos_mean']:.6f} m\\n")
            f.write(f"  Orientation: {stats['error_ori_mean']:.6f} rad\\n")
            f.write(f"  Gripper: {stats['error_grip_mean']:.6f}\\n")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_consistency_analysis(batch_results: Dict, output_dir: Path) -> None:
    """Generate consistency visualization plots"""
    
    stats_list = batch_results['statistics']
    demo_indices = batch_results['demo_indices']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract statistics for plotting
    mean_errors = [s['mean_error'] for s in stats_list]
    std_errors = [s['std_error'] for s in stats_list]
    max_errors = [s['max_error'] for s in stats_list]
    percentile_95 = [s['percentile_95'] for s in stats_list]
    
    # Plot 1: Mean error per demo with error bars
    axes[0, 0].errorbar(demo_indices, mean_errors, yerr=std_errors, fmt='o-',
                        linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[0, 0].axhline(0.1, color='r', linestyle='--', linewidth=2, label='Target: 0.1 m')
    axes[0, 0].set_xlabel('Demo Index', fontsize=11)
    axes[0, 0].set_ylabel('Mean Error (m)', fontsize=11)
    axes[0, 0].set_title('Mean Error per Demo (with std dev)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Max error per demo
    axes[0, 1].bar(demo_indices, max_errors, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(0.15, color='r', linestyle='--', linewidth=2, label='Concern threshold: 0.15 m')
    axes[0, 1].set_xlabel('Demo Index', fontsize=11)
    axes[0, 1].set_ylabel('Max Error (m)', fontsize=11)
    axes[0, 1].set_title('Maximum Error per Demo', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].legend()
    
    # Plot 3: Error distribution (box plot)
    error_pcts = [s['error_as_pct'] for s in stats_list]
    axes[1, 0].bar(demo_indices, error_pcts, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 0].set_xlabel('Demo Index', fontsize=11)
    axes[1, 0].set_ylabel('Error (% of state norm)', fontsize=11)
    axes[1, 0].set_title('Error as Percentage of State Norm', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: 95th percentile
    axes[1, 1].plot(demo_indices, percentile_95, marker='s', linewidth=2,
                    markersize=8, color='darkgreen')
    axes[1, 1].axhline(0.1, color='r', linestyle='--', linewidth=2, label='Target: 0.1 m')
    axes[1, 1].set_xlabel('Demo Index', fontsize=11)
    axes[1, 1].set_ylabel('95th Percentile Error (m)', fontsize=11)
    axes[1, 1].set_title('95th Percentile Error per Demo', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    consistency_file = output_dir / "libero_inverse_action_batch_consistency.png"
    plt.savefig(consistency_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved consistency plot to: {consistency_file}")


def plot_error_heatmap(batch_results: Dict, output_dir: Path) -> None:
    """Generate heatmap of errors across demos and steps"""
    
    # Pad errors to same length for heatmap
    errors_by_demo = batch_results['errors_by_demo']
    demo_indices = batch_results['demo_indices']
    
    max_steps = max(len(e) for e in errors_by_demo)
    heatmap_data = np.zeros((len(errors_by_demo), max_steps))
    
    for i, errors in enumerate(errors_by_demo):
        heatmap_data[i, :len(errors)] = errors
    
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.15)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Demo Index', fontsize=12)
    ax.set_yticks(range(len(demo_indices)))
    ax.set_yticklabels(demo_indices)
    ax.set_title('Inverse Action Error Heatmap (Demo × Step)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error (m)', fontsize=11)
    
    plt.tight_layout()
    heatmap_file = output_dir / "libero_inverse_action_batch_heatmap.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved heatmap to: {heatmap_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch validation of inverse action reversibility across multiple demos"
    )
    parser.add_argument("--demo_start", type=int, default=1,
                        help="Starting demo index (default: 1)")
    parser.add_argument("--demo_end", type=int, default=20,
                        help="Ending demo index (default: 20)")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of steps per demo (default: 50)")
    args = parser.parse_args()
    
    print("\\n" + "=" * 80)
    print("LIBERO BATCH INVERSE ACTION VALIDATION")
    print("=" * 80)
    
    # Run batch test
    batch_results = test_batch(args.demo_start, args.demo_end, args.num_steps)
    
    if len(batch_results['statistics']) == 0:
        print("\\nERROR: No demos successfully tested!")
        return
    
    print("\\n" + "=" * 80)
    print("BATCH RESULTS SUMMARY")
    print("=" * 80)
    
    stats_list = batch_results['statistics']
    mean_errors = [s['mean_error'] for s in stats_list]
    
    print(f"\\nDemos tested: {len(batch_results['demo_indices'])}")
    print(f"  Average mean error: {np.mean(mean_errors):.6f} m")
    print(f"  Std dev: {np.std(mean_errors):.6f} m")
    print(f"  Range: {np.min(mean_errors):.6f} - {np.max(mean_errors):.6f} m")
    
    # Save results
    print("\\nSaving results...")
    save_batch_results(batch_results, OUTPUT_DIR)
    plot_consistency_analysis(batch_results, OUTPUT_DIR)
    plot_error_heatmap(batch_results, OUTPUT_DIR)
    
    print("\\n" + "=" * 80)
    print("✓ BATCH VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
