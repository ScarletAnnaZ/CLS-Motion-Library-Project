import os
import numpy as np
from bvh import Bvh

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')  # original BVH data
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'processed_bvhtest')  # output

TARGET_FRAMES = 600  # each motion‘s expected frames

def read_bvh(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = file.read()
        return Bvh(data), data

# Interpolate frame data to match target frame count
def interpolate_frames(original_frames, target_frames):
    original_array = np.array(original_frames, dtype=float)
    original_len = len(original_array)
    new_indices = np.linspace(0, original_len - 1, target_frames)

    # Apply interpolation for each channel across frames
    interpolated = np.array([
        np.interp(new_indices, np.arange(original_len), original_array[:, i])
        for i in range(original_array.shape[1])
    ]).T

    return interpolated.tolist()

# Resample or truncate a BVH motion to a fixed number of frames
def resample_to_fixed_length(bvh_data, original_data, target_frames):
    original_frames = len(bvh_data.frames)
    frame_time = bvh_data.frame_time
    header, rest = original_data.split("Frame Time:", 1)
    header = header.strip()
    
    
    if original_frames >= target_frames:
         # Truncate if motion is longer than target
        selected = bvh_data.frames[:target_frames]
        resampled_frames = [[f"{value}" for value in frame] for frame in selected]
    else:
        # Interpolate if motion is shorter than target
        interpolated = interpolate_frames(bvh_data.frames, target_frames)
        resampled_frames = [[f"{value:.6f}" for value in frame] for frame in interpolated]

    # Recombine the BVH text content
    new_frame_data = "\n".join([" ".join(frame) for frame in resampled_frames])
    bvh_text = f"{header}\nFrame Time: {frame_time:.6f}\n{new_frame_data}"

    return bvh_text

def save_bvh(bvh_text, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(bvh_text)

# Process all BVH files
def process_bvh_files(data_dir, output_dir):
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.bvh'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, data_dir)
                output_path = os.path.join(output_dir, relative_path)

                try:
                    bvh_data, original_data = read_bvh(input_path)
                    processed_text = resample_to_fixed_length(bvh_data, original_data, TARGET_FRAMES)
                    save_bvh(processed_text, output_path)
                    print(f"✅ Processed: {relative_path}")
                except Exception as e:
                    print(f"❌ Failed: {relative_path} | Error: {e}")

if __name__ == "__main__":
    process_bvh_files(DATA_DIR, OUTPUT_DIR)
