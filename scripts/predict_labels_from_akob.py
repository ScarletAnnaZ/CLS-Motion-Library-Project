import os
from bvh import Bvh

# 项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 指定你想处理的文件路径（直接到具体文件）
AKOB_BVH_FILE = os.path.join(BASE_DIR, 'input_AKOB', '1stmay', 'Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh')  # ❗ 手动指定到单个.bvh文件

# read BVH
def read_bvh(filepath):
    with open(filepath, 'r') as f:
        bvh = Bvh(f.read())
    return bvh

def main():
    bvh = read_bvh(AKOB_BVH_FILE)
    
    total_frames = len(bvh.frames)
    frame_time = bvh.frame_time
    
    print(f"✅ Loaded BVH: {AKOB_BVH_FILE}")
    print(f"Total frames: {total_frames}")
    print(f"Frame time: {frame_time} seconds per frame")

import numpy as np



# change the frame rate to 120fps
def resample_frames(frames, source_fps=100, target_fps=120):
    """
    对 bvh frames 做插值，把 source_fps 帧率提升到 target_fps。
    frames: np.array, shape=(原帧数, 通道数)
    返回: 新的插值后的 frames
    """
    original_frame_count = frames.shape[0]
    feature_dim = frames.shape[1]

    # 计算新的目标帧数
    target_frame_count = int(original_frame_count * (target_fps / source_fps))

    # 原始时间点（比如0,1,2,...）
    original_times = np.linspace(0, original_frame_count-1, original_frame_count)
    # 目标时间点（密集）
    target_times = np.linspace(0, original_frame_count-1, target_frame_count)

    # 对每一列特征单独插值
    resampled = []
    for i in range(feature_dim):
        original_series = frames[:, i]
        interpolated_series = np.interp(target_times, original_times, original_series)
        resampled.append(interpolated_series)

    resampled = np.stack(resampled, axis=1)  # (新帧数, 特征数)
    return resampled


if __name__ == "__main__":
    main()
