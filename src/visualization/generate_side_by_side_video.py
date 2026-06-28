import os
import sys
import h5py
import numpy as np
import imageio
from pathlib import Path
import json

sys.path.append(str(Path("/home/dhruv/Trajectory_Augmentation/src/third_party/LIBERO").resolve()))

from robosuite import make
import libero.libero.envs

def main():
    print("Generating side-by-side video of original vs augmented trajectory...")
    
    orig_path = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    aug_path = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets-augmented/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    
    os.makedirs("videos", exist_ok=True)
    
    with h5py.File(orig_path, 'r') as f_orig, h5py.File(aug_path, 'r') as f_aug:
        # Find a demo that has a decent length (e.g., t is small so augmented length is long)
        aug_demo = "demo_0"
        for k in f_aug['data'].keys():
            aug_len = len(f_aug[f'data/{k}/actions'])
            if aug_len > 100:
                aug_demo = k
                break
                
        # Since we generate 5 augmentations per trajectory, demo_{idx} in augmented
        # corresponds to demo_{idx // 5} in original.
        orig_demo = f"demo_{int(aug_demo.split('_')[1]) // 5}"
        
        print(f"Selected Augmented Demo: {aug_demo}")
        print(f"Corresponding Original Demo: {orig_demo}")
        
        orig_states = f_orig[f'data/{orig_demo}/states'][:]
        aug_states = f_aug[f'data/{aug_demo}/states'][:]
        
        env_args = json.loads(f_orig['data'].attrs['env_args'])
        env_name = env_args.get("env_name", "Libero_Tabletop_Manipulation")
        
        print(f"Original Length: {len(orig_states)}")
        print(f"Augmented Length: {len(aug_states)}")
        
        # Calculate t offset
        t = len(orig_states) - len(aug_states) + 1
        print(f"Calculated offset t: {t}")
        
    # Setup environment for rendering
    bddl_path = "/home/dhruv/Trajectory_Augmentation/src/third_party/LIBERO/libero/libero/bddl_files/libero_goal/put_the_wine_bottle_on_the_rack.bddl"
    
    env = make(
        env_name=env_name,
        bddl_file_name=bddl_path,
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=256,
        camera_widths=256,
    )
    
    env.reset()
    
    def render_state(state):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        # Use robosuite render function directly
        img = env.sim.render(camera_name="agentview", height=256, width=256)[0]
        # Robosuite returns upside down image
        img = np.flipud(img)
        return img
        
    print("Rendering original frames...")
    orig_frames = []
    for s in orig_states:
        orig_frames.append(render_state(s))
        
    print("Rendering augmented frames...")
    aug_frames = []
    # Pad the beginning so they sync up
    first_aug_frame = render_state(aug_states[0])
    for _ in range(t - 1):
        aug_frames.append(first_aug_frame)
        
    for s in aug_states:
        aug_frames.append(render_state(s))
        
    # Pad the end just in case there's a 1 frame diff
    while len(aug_frames) < len(orig_frames):
        aug_frames.append(aug_frames[-1])
    while len(orig_frames) < len(aug_frames):
        orig_frames.append(orig_frames[-1])
        
    print("Stitching side-by-side...")
    side_by_side = []
    import cv2
    for i in range(len(orig_frames)):
        # Stitch horizontally
        combined = np.hstack((orig_frames[i], aug_frames[i]))
        
        # Add text labels using cv2
        combined = np.ascontiguousarray(combined)
        cv2.putText(combined, "Original Trajectory", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "Augmented (Noisy Start + Recovery)", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        side_by_side.append(combined)
        
    print(f"Frame shape before saving: {side_by_side[0].shape}")
    
    out_path = "/home/dhruv/Trajectory_Augmentation/videos/side_by_side_comparison.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    import imageio
    uint8_frames = [f.astype(np.uint8) for f in side_by_side]
    imageio.mimsave(out_path, uint8_frames, fps=20, format='FFMPEG', macro_block_size=None)
    
    print(f"Saved side-by-side video to {out_path}")

if __name__ == "__main__":
    main()
