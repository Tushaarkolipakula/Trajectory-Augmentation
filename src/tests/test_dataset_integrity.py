#!/usr/bin/env python3
"""
Test Dataset Integrity
Consolidates checks for loading the dataset in LeRobot schema, 
comparing features against baselines, and smoke testing cloud configurations.
"""

import sys
import torch
import traceback
import argparse

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Warning: Failed to import standard LeRobotDataset. Attempting fallback.")
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"Failed to import LeRobotDataset: {e}")
        LeRobotDataset = None


def test_dataset_load(dataset_path: str) -> None:
    """
    Attempts to instantiate the local dataset and load the first frame.
    
    Args:
        dataset_path: Absolute path to the LeRobot dataset.
    """
    if not LeRobotDataset:
        print("[Skip] LeRobotDataset not available.")
        return

    print(f"\\n--- Testing Local Dataset Load: {dataset_path} ---")
    try:
        dataset = LeRobotDataset(repo_id=None, root=dataset_path)
        print("  ✓ Dataset successfully loaded!")
        print(f"  Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}")
        print(f"  Features: {list(dataset.features.keys())}")
        
        # Test frame access
        _ = dataset[0]
        print("  ✓ Successfully loaded frame 0.")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        traceback.print_exc()


def compare_datasets(baseline_path: str, target_path: str) -> None:
    """
    Compares the schema and feature shapes between a baseline and a target dataset.
    
    Args:
        baseline_path: Path to the reference dataset.
        target_path: Path to the new/augmented dataset.
    """
    if not LeRobotDataset:
        return

    print(f"\\n--- Comparing Datasets ---")
    
    def print_info(name, path):
        print(f"\\n[{name}]")
        try:
            ds = LeRobotDataset(repo_id=None, root=path)
            print(f"  Total episodes: {ds.num_episodes}")
            print(f"  Total frames: {ds.num_frames}")
            print(f"  FPS: {ds.fps}")
            
            item = ds[0]
            print("  Feature Shapes:")
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape} ({v.dtype})")
                else:
                    print(f"    {k}: {type(v)}")
        except Exception as e:
            print(f"  Error loading {name}: {e}")

    print_info("Baseline (e.g. NVIDIA)", baseline_path)
    print_info("Target (e.g. Augmented)", target_path)


def smoke_test_hf_dataset(repo_id: str) -> None:
    """
    Smoke tests a HuggingFace Hub dataset by loading its schema.
    
    Args:
        repo_id: The HuggingFace repository ID (e.g. 'Sylvest/libero_plus_lerobot').
    """
    if not LeRobotDataset:
        return

    print(f"\\n--- Smoke Test HuggingFace Hub: {repo_id} ---")
    try:
        ds = LeRobotDataset(repo_id)
        print("  ✓ SMOKE TEST PASSED!")
        print(f"  Episodes: {ds.num_episodes}, Frames: {ds.num_frames}")
        print(f"  Features: {list(ds.features.keys())}")
    except Exception as e:
        print(f"  ✗ SMOKE TEST FAILED: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Dataset Integrity")
    parser.add_argument("--local_path", type=str, default="/home/dhruv/Trajectory_Augmentation/data/lerobot_format/libero_goal_augmented", help="Path to local dataset")
    parser.add_argument("--baseline_path", type=str, default="/home/dhruv/Trajectory_Augmentation/data/nvidia_libero/libero_goal", help="Path to baseline dataset")
    parser.add_argument("--hf_repo", type=str, default="Sylvest/libero_plus_lerobot", help="HuggingFace repo ID")
    
    args = parser.parse_args()
    
    test_dataset_load(args.local_path)
    compare_datasets(args.baseline_path, args.local_path)
    smoke_test_hf_dataset(args.hf_repo)
