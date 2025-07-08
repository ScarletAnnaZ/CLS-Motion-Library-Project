import os
import json
import numpy as np
import pandas as pd
from bvh import Bvh


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'output', 'processed600_bvh')
LABEL_FILE = os.path.join(BASE_DIR, 'output', 'standardized_labels.json')
OUTPUT_FILE = os.path.join(BASE_DIR, 'output', 'features', 'full_frame_features.csv')

# The channel you want to extract
POSITION_CHANNELS = ['Xposition', 'Yposition', 'Zposition']
ROTATION_CHANNELS = ['Zrotation', 'Xrotation', 'Yrotation']
ALL_CHANNELS = POSITION_CHANNELS + ROTATION_CHANNELS

def extract_bvh_features(filepath):
    with open(filepath, 'r') as f:
        bvh = Bvh(f.read())

    joint_channels = {}
    for joint in bvh.get_joints():
        joint_name = joint.name
        channels = bvh.joint_channels(joint_name)
        if any(c in ALL_CHANNELS for c in channels):
            joint_channels[joint.name] = [c for c in channels if c in ALL_CHANNELS]

    all_frames = []
    for i in range(len(bvh.frames)):
        frame_data = []
        for joint, channels in joint_channels.items():
            for ch in channels:
                try:
                    val = float(bvh.frame_joint_channel(i, joint, ch))
                except:
                    val = 0.0
                frame_data.append(val)
        all_frames.append(frame_data)

    arr = np.array(all_frames)  # shape = (600, num_features)
    return {
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0),
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
    }

def build_feature_vector(stat_dict):
    vec = []
    for stat in ['mean', 'std', 'min', 'max']:
        vec.extend(stat_dict[stat])
    return vec

def main():
    with open(LABEL_FILE, 'r') as f:
        labels = json.load(f)

    all_vectors = []
    all_ids = []
    all_cats = []

    for motion_id, info in labels.items():
        if not motion_id.endswith('.bvh'):
            motion_id += '.bvh'
        folder = motion_id.split('_')[0]
        full_path = os.path.join(DATA_DIR, folder, motion_id)

        if not os.path.exists(full_path):
            print(f"❌ Not found: {full_path}")
            continue

        try:
            stats = extract_bvh_features(full_path)
            vector = build_feature_vector(stats)
            all_vectors.append(vector)
            all_ids.append(motion_id)
            all_cats.append(info['description'])
            print(f"✅ Processed: {motion_id}")
        except Exception as e:
            print(f"❌ Error {motion_id}: {e}")

    df = pd.DataFrame(all_vectors)
    df['motion_id'] = all_ids
    df['Label'] = all_cats

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
