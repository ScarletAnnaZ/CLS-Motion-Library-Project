import os
import json
import numpy as np
import pandas as pd
from bvh import Bvh

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'output', 'processed600_bvh')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'features')
LABELS_FILE = os.path.join(BASE_DIR, 'output', 'standardized_labels.json')
SUMMARY_FILE = os.path.join(BASE_DIR, 'output', 'small_bvh_summary.csv')
TARGET_FRAMES = 600

def read_bvh(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
        return Bvh(data)

def extract_features(bvh_data):
    """
    从 BVH 文件中提取特征向量（位置 + 旋转）
    返回：一个 (帧数, 特征数) 的矩阵
    """
    frames = np.array(bvh_data.frames, dtype=float)
    return frames  # 返回每一帧的所有数据作为特征矩阵

def load_labels(labels_file):
    with open(labels_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_file(file_path, label):
    bvh_data = read_bvh(file_path)
    features = extract_features(bvh_data)
    labels = [label] * len(features)  # 每一帧都附上相同的标签
    return features, labels

def save_features(features_list, labels_list, output_file):
    # 将所有特征与标签合并
    all_features = np.vstack(features_list)
    all_labels = np.concatenate(labels_list)

    # 将数据保存为 DataFrame
    df = pd.DataFrame(all_features)
    df['Label'] = all_labels  # 添加标签列
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Features saved to {output_file}")

def process_all_files(summary_file, labels_dict, processed_dir, output_file):
    summary_df = pd.read_csv(summary_file)
    features_list = []
    labels_list = []

    for index, row in summary_df.iterrows():
        full_motion_id = row['File']  # 原始格式包含路径信息 e.g., "132/132_52.bvh"
        motion_id = os.path.basename(full_motion_id).replace(".bvh", "")  # 
        
        if motion_id in labels_dict:
            label = labels_dict[motion_id]["description"]
            relative_path = os.path.join(processed_dir, full_motion_id)
            
            if os.path.exists(relative_path):
                try:
                    features, labels = process_file(relative_path, label)
                    features_list.append(features)
                    labels_list.append(labels)
                    print(f"✅ Processed: {motion_id}")
                except Exception as e:
                    print(f"❌ Failed to process {motion_id}: {e}")
            else:
                print(f"❌ File not found: {relative_path}")
        else:
            print(f"❌ Label not found for {motion_id}")

    save_features(features_list, labels_list, output_file)

if __name__ == "__main__":
    labels_dict = load_labels(LABELS_FILE)
    output_file = os.path.join(OUTPUT_DIR, 'small_features.csv')
    process_all_files(SUMMARY_FILE, labels_dict, PROCESSED_DIR, output_file)
