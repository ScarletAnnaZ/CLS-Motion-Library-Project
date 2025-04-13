import os
import pandas as pd
import numpy as np
from bvh import Bvh

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed_bvh')
INPUT_CSV = os.path.join(OUTPUT_DIR, 'short_bvh_files.csv')
TARGET_FRAMES = 600

def read_bvh(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
        return Bvh(data), data

def interpolate_bvh(bvh_data, original_data, target_frames):
    original_frames = len(bvh_data.frames)
    if original_frames >= target_frames:
        return original_data  # 如果已经超过目标帧数，跳过

    # 获取原始帧数据
    frame_data = np.array(bvh_data.frames, dtype=float)

    # 获取数据维度（每一帧的骨架数据数量）
    num_joints = frame_data.shape[1]

    # 原始帧索引
    original_indices = np.arange(original_frames)

    # 目标帧索引
    target_indices = np.linspace(0, original_frames - 1, target_frames)

    # 对每一个关节的数据进行插值
    interpolated_frames = []
    for joint_index in range(num_joints):
        joint_data = frame_data[:, joint_index]
        interpolated_joint_data = np.interp(target_indices, original_indices, joint_data)
        interpolated_frames.append(interpolated_joint_data)

    # 转换成 (帧数, 关节数) 的格式
    interpolated_frames = np.array(interpolated_frames).T

    # 修改原始数据中的帧列表
    bvh_data.frames = interpolated_frames.tolist()

    # 修改文件头部信息
    processed_data = original_data.replace(f"Frames: {original_frames}", f"Frames: {target_frames}")
    processed_data = processed_data.replace(f"Frame Time: {bvh_data.frame_time}", f"Frame Time: {1 / 120}")

    # 添加插值后的数据行
    motion_data = ""
    for frame in interpolated_frames:
        frame_line = " ".join([f"{v:.6f}" for v in frame])
        motion_data += frame_line + "\n"
        
    processed_data = processed_data.split("MOTION")[0] + "MOTION\nFrames: " + str(target_frames) + "\nFrame Time: " + str(1 / 120) + "\n" + motion_data

    return processed_data

def save_bvh(processed_data, output_path):
    with open(output_path, 'w') as file:
        file.write(processed_data)

def process_files(csv_file, data_dir, processed_dir):
    short_files = pd.read_csv(csv_file)
    
    for index, row in short_files.iterrows():
        relative_path = row['File']
        file_path = os.path.join(data_dir, relative_path)
        output_path = os.path.join(processed_dir, relative_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # 读取原始 BVH 文件
            bvh_data, original_data = read_bvh(file_path)
            
            # 插值处理
            processed_data = interpolate_bvh(bvh_data, original_data, TARGET_FRAMES)
            
            # 保存处理后的文件
            save_bvh(processed_data, output_path)
            
            print(f"✅ Processed: {relative_path}")
        
        except Exception as e:
            print(f"❌ Failed to process {relative_path}: {e}")

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    process_files(INPUT_CSV, DATA_DIR, PROCESSED_DIR)
