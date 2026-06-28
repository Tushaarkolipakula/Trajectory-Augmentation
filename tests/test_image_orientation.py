#!/usr/bin/env python3
import h5py
import imageio
import numpy as np
from pathlib import Path

def test_hdf5_orientation(filepath: str, demo_idx: int = 0) -> None:
    """
    Checks the orientation of the agentview RGB images inside an HDF5 file.
    It compares the average pixel brightness of the top vs bottom rows.

    Args:
        filepath: Absolute path to the HDF5 dataset file.
        demo_idx: Index of the demo to check.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[Skip] File not found: {filepath}")
        return

    print(f"\\n--- Testing HDF5 Orientation: {filepath.name} ---")
    with h5py.File(filepath, 'r') as f:
        frames = f[f'data/demo_{demo_idx}/obs/agentview_rgb'][:]
        print(f"Total frames: {len(frames)}")
        
        # Test first, second, and last frame
        for i in [0, 1, len(frames)-1]:
            frame = frames[i]
            top_avg = np.mean(frame[:10, :, :])
            bottom_avg = np.mean(frame[-10:, :, :])
            
            if top_avg > bottom_avg:
                print(f"  Frame {i}: RIGHT-SIDE UP (Top: {top_avg:.1f}, Bottom: {bottom_avg:.1f})")
            else:
                print(f"  Frame {i}: UPSIDE DOWN (Top: {top_avg:.1f}, Bottom: {bottom_avg:.1f})")

def test_environment_orientation(suite_name: str = "libero_object", task_id: int = 0) -> None:
    """
    Checks the native image orientation generated directly by the Libero simulation environment.

    Args:
        suite_name: The name of the libero suite to instantiate.
        task_id: The index of the task.
    """
    try:
        from lerobot.envs.libero import _get_suite, LiberoEnv
    except ImportError:
        print("[Skip] lerobot not installed. Skipping environment test.")
        return

    print(f"\\n--- Testing Environment Orientation: {suite_name} (Task {task_id}) ---")
    suite = _get_suite(suite_name)
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        camera_name="agentview_image",
    )
    obs, _ = env.reset()
    img = obs["pixels"]["image"]

    top_avg = np.mean(img[:10, :, :])
    bottom_avg = np.mean(img[-10:, :, :])

    if top_avg > bottom_avg:
        print(f"  Env native output: RIGHT-SIDE UP (Top: {top_avg:.1f}, Bottom: {bottom_avg:.1f})")
    else:
        print(f"  Env native output: UPSIDE DOWN (Top: {top_avg:.1f}, Bottom: {bottom_avg:.1f})")

def export_test_video(filepath: str, output_mp4: str) -> None:
    """
    Exports the agentview RGB stream from an HDF5 demo to an MP4 video for manual inspection.

    Args:
        filepath: Absolute path to the HDF5 dataset file.
        output_mp4: Absolute path to save the output MP4 video.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[Skip] File not found: {filepath}")
        return

    print(f"\\n--- Exporting Video: {filepath.name} ---")
    with h5py.File(filepath, 'r') as f:
        frames = f['data/demo_0/obs/agentview_rgb'][:]
        imageio.mimsave(output_mp4, frames, fps=20, format='FFMPEG')
    print(f"  Video saved to: {output_mp4}")

if __name__ == "__main__":
    orig_file = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    aug_file = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets-augmented/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    video_out = "/home/dhruv/Trajectory_Augmentation/videos/baseline_test.mp4"
    
    test_hdf5_orientation(orig_file)
    test_hdf5_orientation(aug_file)
    test_environment_orientation()
    # export_test_video(orig_file, video_out)
