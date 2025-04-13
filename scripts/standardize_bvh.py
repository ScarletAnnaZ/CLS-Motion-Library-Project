import os
import numpy as np
from bvh import Bvh

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# normalizing parameter
TARGET_FPS = 120
TARGET_FRAMES = 600  # 5 seconds of motion

def read_bvh(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
        return Bvh(data), data

def resample_bvh(bvh_data, original_data, target_fps, target_frames):
    original_fps = 1.0 / float(bvh_data.frame_time)
    original_frames = len(bvh_data.frames)
    
    # 计算缩放因子
    scaling_factor = target_fps / original_fps
    new_frame_count = int(original_frames * scaling_factor)
    
    # 对帧进行插值
    interpolated_frames = np.linspace(0, original_frames - 1, num=new_frame_count)
    resampled_data = [bvh_data.frames[int(i)] for i in interpolated_frames]
    
    # 裁剪或填充至目标帧数
    if len(resampled_data) > target_frames:
        resampled_data = resampled_data[:target_frames]
    else:
        while len(resampled_data) < target_frames:
            resampled_data.append(resampled_data[-1])
    
    # update BVH 数据的帧数与帧时间
    frame_time = 1.0 / target_fps
    processed_data = original_data.replace(f"Frames: {original_frames}", f"Frames: {target_frames}")
    processed_data = processed_data.replace(f"Frame Time: {bvh_data.frame_time}", f"Frame Time: {frame_time}")
    
    bvh_data.frames = resampled_data
    return processed_data

def save_bvh(bvh_data, output_path):
    with open(output_path, 'w') as file:
        file.write(bvh_data)

def process_bvh_files(data_dir, output_dir):
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.bvh'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, data_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                # create and output file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    # read BVH
                    bvh_data, original_data = read_bvh(input_path)
                    
                    # normalization
                    processed_bvh = resample_bvh(bvh_data, original_data, TARGET_FPS, TARGET_FRAMES)
                    
                    # store the file 
                    save_bvh(processed_bvh, output_path)
                    
                    print(f"✅ Processed: {relative_path}")
                
                except Exception as e:
                    print(f"❌ Failed to process {relative_path}: {e}")

if __name__ == "__main__":
    process_bvh_files(DATA_DIR, OUTPUT_DIR)
