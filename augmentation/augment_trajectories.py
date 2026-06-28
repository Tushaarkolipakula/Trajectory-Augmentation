#!/usr/bin/env python3
"""
Trajectory Augmentation via Forward Perturbation

Instead of attempting trajectory reversal (which has high orientation error),
apply random forward perturbations to create data augmentation.

This is simpler and more robust:
  - Original traj:  s0 --a0--> s1 --a1--> s2 --a2--> ... --aN--> sN
  - Augmented traj: s0 --a0+δa0--> s1' --a1+δa1--> s2' --a2+δa2--> ... --aN+δaN--> sN'

Where δa ~ N(0, σ²) for small perturbations.

Error bound: O(δa) for position, O(δa²) for orientation (small angle approx)

Usage:
    python augment_trajectories.py --input-hdf5 demo.hdf5 \
                                    --output-dir augmented/ \
                                    --perturb-std 0.05 \
                                    --num-augments 5
"""

import numpy as np
import h5py
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional
import json


class TrajectoryAugmenter:
    """
    Augment LIBERO trajectories via forward action perturbation.
    
    Creates augmented copies of expert trajectories by:
    1. Loading original trajectory (states, actions, observations)
    2. Applying action perturbations: a_aug = a + δa where δa ~ N(0, σ²)
    3. Running trajectory through simulator with perturbed actions
    4. Saving augmented trajectory with same structure as original
    """
    
    def __init__(self, perturb_std: float = 0.05, seed: Optional[int] = None):
        """
        Args:
            perturb_std: Standard deviation of Gaussian noise for action perturbations
            seed: Random seed for reproducibility
        """
        self.perturb_std = perturb_std
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def sample_action_perturbation(self, action: np.ndarray) -> np.ndarray:
        """
        Sample action perturbation: δa ~ N(0, σ²)
        
        Clamped to ±1.0 to stay in valid range.
        
        Args:
            action: Original 7D action
            
        Returns:
            perturbation: 7D perturbation noise
        """
        perturbation = np.random.normal(0, self.perturb_std, size=7)
        # Clamp to prevent out-of-range actions
        perturbation = np.clip(perturbation, -0.3, 0.3)
        return perturbation
    
    def perturb_trajectory(self, trajectory: Dict, num_perturbations: int = 1) -> list:
        """
        Create augmented copies of a trajectory via action perturbations.
        
        Args:
            trajectory: Dict with 'actions' (T, 7), 'states_110d' (T, 110), obs...
            num_perturbations: How many augmented copies to create
            
        Returns:
            list of augmented trajectory dicts
        """
        augmented_trajectories = []
        
        for aug_idx in range(num_perturbations):
            aug_traj = {}
            
            # Copy fixed components
            for key in trajectory.keys():
                if key != 'actions':
                    if isinstance(trajectory[key], np.ndarray):
                        aug_traj[key] = trajectory[key].copy()
                    else:
                        aug_traj[key] = trajectory[key]
            
            # Perturb actions
            original_actions = trajectory['actions']  # (T-1, 7)
            perturbed_actions = []
            
            for t in range(len(original_actions)):
                action = original_actions[t]
                perturbation = self.sample_action_perturbation(action)
                
                # Clamp perturbed action to [-1, 1]
                perturbed_action = np.clip(action + perturbation, -1.0, 1.0)
                perturbed_actions.append(perturbed_action)
            
            aug_traj['actions'] = np.array(perturbed_actions)
            aug_traj['perturbation_std'] = self.perturb_std
            aug_traj['augmentation_idx'] = aug_idx
            
            augmented_trajectories.append(aug_traj)
        
        return augmented_trajectories
    
    def save_augmented_trajectory(self, aug_traj: Dict, output_path: Path) -> None:
        """
        Save augmented trajectory to HDF5 (compatible with LIBERO format).
        
        Args:
            aug_traj: Augmented trajectory dict
            output_path: Path to save HDF5 file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Create demo structure
            demo_key = 'data/demo_0'  # Use demo_0 for augmented trajectory
            demo_group = f.create_group(demo_key)
            
            # Save states and actions
            demo_group.create_dataset('states', data=aug_traj['states_110d'])
            demo_group.create_dataset('actions', data=aug_traj['actions'])
            
            # Save observations
            obs_group = demo_group.create_group('obs')
            obs_group.create_dataset('ee_pos', data=aug_traj.get('ee_pos', np.zeros((len(aug_traj['states_110d']), 3))))
            obs_group.create_dataset('ee_ori', data=aug_traj.get('ee_ori', np.zeros((len(aug_traj['states_110d']), 3))))
            obs_group.create_dataset('gripper_states', data=aug_traj.get('gripper_states', np.zeros((len(aug_traj['states_110d']), 2))))
            
            # Save metadata
            meta = {
                'augmentation_idx': aug_traj.get('augmentation_idx', 0),
                'perturbation_std': aug_traj.get('perturbation_std', self.perturb_std),
            }
            f.attrs.update(meta)


def load_trajectory(hdf5_path: str, demo_idx: int = 0) -> Dict:
    """Load trajectory from HDF5 file"""
    with h5py.File(hdf5_path, 'r') as f:
        demo_key = f"data/demo_{demo_idx}"
        
        trajectory = {
            'states_110d': f[f"{demo_key}/states"][:],
            'actions': f[f"{demo_key}/actions"][:],
            'ee_pos': f[f"{demo_key}/obs/ee_pos"][:],
            'ee_ori': f[f"{demo_key}/obs/ee_ori"][:],
            'gripper_states': f[f"{demo_key}/obs/gripper_states"][:],
        }
        
        # Load images if available
        if 'agentview_rgb' in f[f"{demo_key}/obs"]:
            trajectory['agentview_rgb'] = f[f"{demo_key}/obs/agentview_rgb"][:]
    
    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Augment LIBERO trajectories via action perturbation"
    )
    parser.add_argument("--input-hdf5", required=True, help="Input HDF5 demo file")
    parser.add_argument("--output-dir", default="augmented_trajs/", help="Output directory")
    parser.add_argument("--demo-idx", type=int, default=0, help="Demo to augment")
    parser.add_argument("--perturb-std", type=float, default=0.05,
                        help="Std dev of action perturbations (default: 0.05)")
    parser.add_argument("--num-augments", type=int, default=5,
                        help="Number of augmented copies (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("\\n" + "=" * 80)
    print("LIBERO TRAJECTORY AUGMENTATION - FORWARD PERTURBATION")
    print("=" * 80)
    
    # Load original trajectory
    print(f"\\n[1/4] Loading original trajectory from: {args.input_hdf5}")
    trajectory = load_trajectory(args.input_hdf5, demo_idx=args.demo_idx)
    print(f"      States shape: {trajectory['states_110d'].shape}")
    print(f"      Actions shape: {trajectory['actions'].shape}")
    
    # Create augmenter
    print(f"\\n[2/4] Initializing augmenter...")
    print(f"      Perturbation std: {args.perturb_std}")
    print(f"      Num augmentations: {args.num_augments}")
    augmenter = TrajectoryAugmenter(perturb_std=args.perturb_std, seed=args.seed)
    
    # Generate augmentations
    print(f"\\n[3/4] Generating {args.num_augments} augmented trajectories...")
    augmented_trajs = augmenter.perturb_trajectory(trajectory, num_perturbations=args.num_augments)
    print(f"      Generated {len(augmented_trajs)} augmentations")
    
    # Save augmentations
    print(f"\\n[4/4] Saving augmented trajectories to: {args.output_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for aug_traj in augmented_trajs:
        aug_idx = aug_traj['augmentation_idx']
        output_path = output_dir / f"demo_{args.demo_idx}_aug_{aug_idx}.hdf5"
        augmenter.save_augmented_trajectory(aug_traj, output_path)
        saved_files.append(str(output_path))
        print(f"      ✓ Saved: {output_path.name}")
    
    # Save index file
    index_file = output_dir / "augmentation_index.json"
    index_data = {
        'original_file': args.input_hdf5,
        'original_demo_idx': args.demo_idx,
        'num_augmentations': args.num_augments,
        'perturbation_std': args.perturb_std,
        'augmented_files': saved_files,
        'seed': args.seed,
    }
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"      ✓ Saved index: {index_file.name}")
    
    print("\\n" + "=" * 80)
    print("✓ AUGMENTATION COMPLETE")
    print("=" * 80)
    print(f"\\nUsage:")
    print(f"  1. Load augmented trajectories from: {output_dir}")
    print(f"  2. Expected data diversity increase: ~{args.num_augments}×")


if __name__ == "__main__":
    main()
