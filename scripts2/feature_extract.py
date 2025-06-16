import os
import json
import numpy as np
import pandas as pd
from bvh import Bvh

# ===== 路径配置 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BVH_DIR = os.path.join(BASE_DIR, 'output2', 'segments_120')
LABELS_PATH = os.path.join(BASE_DIR, 'output2', 'segment_labels_120.json')
OUTPUT_CSV = os.path.join(BASE_DIR, 'output2', 'segment_features_120.csv')

# ===== 特征提取函数 =====
def extract_features_from_bvh(filepath):
    with open(filepath, 'r') as f:
        mocap = Bvh(f.read())

    frames = np.array(mocap.frames, dtype=float)  # shape: (num_frames, num_channels)

    # 提取统计特征
    means = np.mean(frames, axis=0)
    stds = np.std(frames, axis=0)
    maxs = np.max(frames, axis=0)
    mins = np.min(frames, axis=0)

    # 拼接为特征向量
    features = np.concatenate([means, stds, maxs, mins])
    return features

# ===== 主处理逻辑 =====
def process_all_segments(bvh_dir, labels_path, output_csv):
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)

    data = []
    missing = 0

    for filename in sorted(os.listdir(bvh_dir)):
        if filename.endswith('.bvh'):
            filepath = os.path.join(bvh_dir, filename)

            try:
                features = extract_features_from_bvh(filepath)
                motion_id = filename  # 例如 "001_000.bvh"
                label = label_dict.get(motion_id, None)

                if label is not None:
                    row = {'motion_id': motion_id, 'label': label}
                    for i, val in enumerate(features):
                        row[f'feature_{i}'] = val
                    data.append(row)
                else:
                    print(f"⚠️ Label not found for {motion_id}")
                    missing += 1

            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")
                continue

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Extracted features from {len(data)} files. Missing labels: {missing}")
    print(f"📄 Saved to: {output_csv}")

# ===== 执行入口 =====
if __name__ == "__main__":
    process_all_segments(BVH_DIR, LABELS_PATH, OUTPUT_CSV)
