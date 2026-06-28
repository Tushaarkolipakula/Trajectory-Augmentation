import h5py
import imageio
import os
import numpy as np

def main():
    aug_path = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets-augmented/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    out_path = "/home/dhruv/Trajectory_Augmentation/videos/augmented_libero_object_traj.mp4"
    
    print(f"Reading from {aug_path}")
    with h5py.File(aug_path, 'r') as f_aug:
        # Get first demo that has a decent length
        demo_key = "demo_0"
        for k in f_aug['data'].keys():
            if len(f_aug[f'data/{k}/actions']) > 50:
                demo_key = k
                break
                
        print(f"Extracting frames from {demo_key}...")
        frames = f_aug[f'data/{demo_key}/obs/agentview_rgb'][:]
        
        # Ensure it's uint8
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
            
        # Flip frames back (since they are stored upside down in the dataset to match LIBERO quirks)
        frames = np.array([np.flipud(frame) for frame in frames])
            
        print(f"Saving {len(frames)} frames to {out_path}...")
        imageio.mimsave(out_path, frames, fps=20, format='FFMPEG', macro_block_size=None)
        
    print("Done!")

if __name__ == "__main__":
    main()
