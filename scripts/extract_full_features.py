import os
import json
import numpy as np
import pandas as pd
from bvh import Bvh

# 项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'output', 'processed600_bvh')
LABEL_FILE = os.path.join(BASE_DIR, 'output', 'standardized_labels.json')
OUTPUT_FILE = os.path.join(BASE_DIR, 'output', 'features', 'full_features.csv')

# 要提取的通道（可扩展）
POSITION_CHANNELS = ['Xposition', 'Yposition', 'Zposition']
ROTATION_CHANNELS = ['Zrotation', 'Xrotation', 'Yrotation']

def extract_bvh_features(filepath, motion_id):
    with open(filepath, 'r') as f:
        bvh = Bvh(f.read())

    joint_channels = {}
    # 收集所有关节的通道（仅位置 + 旋转）
    for joint in bvh.get_joints():
        name = joint.name
        channels = bvh.joint_channels(joint)
        filtered = [ch for ch in channels if ch in POSITION_CHANNELS + ROTATION_CHANNELS]
        if filtered:
            joint_channels[name] = filtered

    # 收集所有帧数据
    all_features = []
    for i in range(len(bvh.frames)):
        frame_data = []
        for joint, channels in joint_channels.items():
            for ch in channels:
                try:
                    val = float(bvh.frame_joint_channel(i, joint, ch))
                    frame_data.append(val)
                except:
                    frame_data.append(0.0)
        all_features.append(frame_data)

    # 转成矩阵并计算统计量
    data = np.array(all_features)  # shape = (600, num_features)
    feature_dict = {
        'motion_id': motion_id,
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0),
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0),
    }
    return feature_dict

def build_feature_vector(feature_dict):
    vec = []
    for stat_name in ['mean', 'std', 'min', 'max']:
        vec.extend(feature_dict[stat_name])
    return vec

def main():
    with open(LABEL_FILE, 'r') as f:
        label_data = json.load(f)

    all_vectors = []
    motion_ids = []
    labels = []

    for motion_id, info in label_data.items():
        filename = motion_id if motion_id.endswith('.bvh') else motion_id + '.bvh'
        rel_path = filename if '/' not in filename else os.path.basename(filename)
        
        full_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(full_path):
            continue
        try:
            feature_dict = extract_bvh_features(full_path, motion_id)
            feature_vector = build_feature_vector(feature_dict)
            all_vectors.append(feature_vector)
            motion_ids.append(motion_id)
            labels.append(info['category'])
            print(f"✅ Extracted features: {motion_id}")
        except Exception as e:
            print(f"❌ Failed: {motion_id} — {e}")
    


    # 生成 DataFrame
    df = pd.DataFrame(all_vectors)
    df['motion_id'] = motion_ids
    df['Label'] = labels
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Features saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
