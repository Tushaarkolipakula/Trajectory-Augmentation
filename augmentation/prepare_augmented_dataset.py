#!/usr/bin/env python3
"""
Integration Guide: Using Augmented LIBERO Trajectories

This script demonstrates how to:
1. Load augmented trajectories
2. Combine with original trajectories
3. Create a unified dataset

Example usage:
    python prepare_augmented_dataset.py \
        --original-hdf5 sandbox/pick_up_the_soup_and_place_it_in_the_basket_demo.hdf5 \
        --augmented-dir /tmp/augmented_demo \
        --output-jsonl dataset_output.jsonl
"""

import h5py
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np


def load_hdf5_trajectory(hdf5_path: str, demo_idx: int = 0) -> Dict:
    """Load a single trajectory from HDF5"""
    with h5py.File(hdf5_path, 'r') as f:
        demo_key = f"data/demo_{demo_idx}"
        
        return {
            'states_110d': f[f"{demo_key}/states"][:],
            'actions': f[f"{demo_key}/actions"][:],
            'ee_pos': f[f"{demo_key}/obs/ee_pos"][:],
            'ee_ori': f[f"{demo_key}/obs/ee_ori"][:],
            'gripper_states': f[f"{demo_key}/obs/gripper_states"][:],
            'agentview_rgb': f[f"{demo_key}/obs/agentview_rgb"][:] if 'agentview_rgb' in f[f"{demo_key}/obs"] else None,
        }


def prepare_training_sample(trajectory: Dict, step_idx: int,  
                           is_augmented: bool = False,
                           aug_idx: int = 0) -> Dict:
    """
    Prepare a single dataset sample (frame + action chunk + observation).
    
    Compatible with LIBERO/VLA dataset format.
    
    Args:
        trajectory: Loaded trajectory dict
        step_idx: Step index in trajectory
        is_augmented: Whether this is from augmented trajectory
        aug_idx: Augmentation index (if augmented)
        
    Returns:
        dataset sample dict
    """
    sample = {
        'step': step_idx,
        'is_augmented': is_augmented,
        'aug_idx': aug_idx if is_augmented else None,
        
        # Current observation
        'state_eef_8d': trajectory['ee_pos'][step_idx].tolist() + \
                        trajectory['ee_ori'][step_idx].tolist() + \
                        trajectory['gripper_states'][step_idx].tolist(),
        
        # Action
        'action': trajectory['actions'][step_idx].tolist(),
        
        # Next observation
        'next_state_eef_8d': trajectory['ee_pos'][step_idx + 1].tolist() + \
                             trajectory['ee_ori'][step_idx + 1].tolist() + \
                             trajectory['gripper_states'][step_idx + 1].tolist() if step_idx + 1 < len(trajectory['actions']) else None,
        
        # Full 110D state for reference
        'state_full_110d': trajectory['states_110d'][step_idx].tolist(),
        'next_state_full_110d': trajectory['states_110d'][step_idx + 1].tolist() if step_idx + 1 < len(trajectory['states_110d']) else None,
    }
    
    # Add image if available
    if trajectory['agentview_rgb'] is not None:
        sample['image'] = trajectory['agentview_rgb'][step_idx].tobytes()  # Binary encoding for JSONL compatibility
        sample['image_shape'] = trajectory['agentview_rgb'][step_idx].shape
    
    return sample


def create_mixed_dataset(original_hdf5: Path, augmented_dir: Path, 
                        output_jsonl: Path, max_trajectories: int = 50) -> None:
    """
    Create a mixed dataset with original + augmented trajectories.
    
    Args:
        original_hdf5: Path to original demo HDF5 file
        augmented_dir: Directory containing augmented HDF5 files
        output_jsonl: Output JSONL file
        max_trajectories: Max number of trajectories to include
    """
    samples = []
    
    # Load original trajectory
    print(f"Loading original trajectory from: {original_hdf5}")
    try:
        original_traj = load_hdf5_trajectory(str(original_hdf5), demo_idx=0)
        print(f"  ✓ Loaded original trajectory: {len(original_traj['actions'])} steps")
        
        # Add samples from original
        for step_idx in range(len(original_traj['actions'])):
            sample = prepare_training_sample(original_traj, step_idx, 
                                            is_augmented=False, aug_idx=0)
            samples.append(sample)
    except Exception as e:
        print(f"  ✗ Failed to load original: {e}")
    
    # Load augmented trajectories
    if augmented_dir.exists():
        augmented_files = sorted(augmented_dir.glob("demo_*_aug_*.hdf5"))
        print(f"\\nLoading {len(augmented_files)} augmented trajectories from: {augmented_dir}")
        
        for aug_file in augmented_files[:max_trajectories]:
            try:
                # Extract augmentation index from filename
                aug_idx = int(aug_file.stem.split('_')[-1])
                
                aug_traj = load_hdf5_trajectory(str(aug_file), demo_idx=0)
                print(f"  ✓ Loaded {aug_file.name}: {len(aug_traj['actions'])} steps")
                
                # Add samples from augmented trajectory
                for step_idx in range(len(aug_traj['actions'])):
                    sample = prepare_training_sample(aug_traj, step_idx,
                                                    is_augmented=True, aug_idx=aug_idx)
                    samples.append(sample)
            except Exception as e:
                print(f"  ✗ Failed to load {aug_file.name}: {e}")
    
    # Save JSONL
    print(f"\\nSaving {len(samples)} samples to: {output_jsonl}")
    with open(output_jsonl, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\\n')
    
    print(f"✓ Created dataset with {len(samples)} samples")
    print(f"  - Original trajectory samples: {len(original_traj['actions'])}")
    print(f"  - Augmented trajectory samples: {len(samples) - len(original_traj['actions'])}")
    print(f"  - Data diversity increase: {len(samples) / len(original_traj['actions']):.1f}×")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mixed dataset (original + augmented)"
    )
    parser.add_argument("--original-hdf5", required=True,
                        help="Path to original HDF5 demo file")
    parser.add_argument("--augmented-dir", required=True,
                        help="Directory with augmented HDF5 files")
    parser.add_argument("--output-jsonl", default="dataset_augmented.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--max-trajectories", type=int, default=50,
                        help="Max number of augmented trajectories to include")
    args = parser.parse_args()
    
    print("\\n" + "=" * 80)
    print("PREPARE AUGMENTED DATASET")
    print("=" * 80)
    
    original_hdf5 = Path(args.original_hdf5)
    augmented_dir = Path(args.augmented_dir)
    output_jsonl = Path(args.output_jsonl)
    
    if not original_hdf5.exists():
        print(f"ERROR: Original HDF5 not found: {original_hdf5}")
        return
    
    if not augmented_dir.exists():
        print(f"ERROR: Augmented directory not found: {augmented_dir}")
        return
    
    create_mixed_dataset(original_hdf5, augmented_dir, output_jsonl,
                        max_trajectories=args.max_trajectories)
    
    print("\\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f\"\"\"
1. Load your newly generated JSONL dataset:
   
   with open('{output_jsonl}') as f:
       samples = [json.loads(line) for line in f]
   
2. Inspect the dataset for physical correctness.

Expected improvements from data augmentation:
   - Better robustness to small physical perturbations
   - More generalized state distributions
\"\"\")


if __name__ == "__main__":
    main()
