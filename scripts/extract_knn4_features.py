import os
import json
import numpy as np
import pandas as pd
from bvh import Bvh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'output', 'processed600_bvh')
LABELS_FILE = os.path.join(BASE_DIR, 'output', 'standardized_labels.json')
OUTPUT_FILE = os.path.join(BASE_DIR, 'output', 'features', 'knn4_features.csv')

# 指定要提取的通道组合（joint, channel, stat_name）
target_channels = [
    ('Hips', 'Yposition', 'mean'),
    ('Head', 'Zrotation', 'std'),
    ('LeftKnee', 'Yrotation', 'std'),
    ('RightElbow', 'Xrotation', 'mean'),
]

def extract_single_value(bvh, joint, channel, stat_type):
    values = []
    for i in range(len(bvh.frames)):
        try:
            val = float(bvh.frame_joint_channel(i, joint, channel))
        except:
            val = 0.0
        values.append(val)

    values = np.array(values)
    if stat_type == 'mean':
        return np.mean(values)
    elif stat_type == 'std':
        return np.std(values)
    else:
        return 0.0

def extract_features(filepath):
    with open(filepath, 'r') as f:
        bvh = Bvh(f.read())

    features = []
    for joint, channel, stat_type in target_channels:
        try:
            val = extract_single_value(bvh, joint, channel, stat_type)
        except:
            val = 0.0
        features.append(val)
    return features

def main():
    with open(LABELS_FILE, 'r') as f:
        label_data = json.load(f)

    all_vectors = []
    all_labels = []
    all_ids = []

    for motion_id, info in label_data.items():
        if not motion_id.endswith('.bvh'):
            motion_id += '.bvh'
        folder = motion_id.split('_')[0]
        path = os.path.join(DATA_DIR, folder, motion_id)

        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue

        try:
            features = extract_features(path)
            all_vectors.append(features)
            all_labels.append(info['category'])
            all_ids.append(motion_id)
            print(f"✅ Processed: {motion_id}")
        except Exception as e:
            print(f"❌ Error {motion_id}: {e}")

    df = pd.DataFrame(all_vectors, columns=[
        'Root_Y_mean',
        'Head_Z_std',
        'LeftKnee_Y_std',
        'RightElbow_X_mean'
    ])
    df['motion_id'] = all_ids
    df['Label'] = all_labels

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
