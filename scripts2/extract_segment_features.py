import os
import json
import numpy as np
import pandas as pd
from bvh import Bvh

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENT_DIR = os.path.join(BASE_DIR, 'output2', 'segments_120')  # The folder of motion with different frames sizes
LABELS_PATH = os.path.join(BASE_DIR, 'output2', 'segment_labels_120.json')
OUTPUT_FILE = os.path.join(BASE_DIR, 'output2', 'features', 'segment120_features.csv')

# Feature extraction: Calculate the mean, std, min, and max of each channel
def extract_statistics_from_segment(bvh_path):
    with open(bvh_path, 'r') as f:
        mocap = Bvh(f.read())
    
    frames = np.array(mocap.frames, dtype=np.float32)  # shape: [T, C] 
    if frames.shape[0] == 0:
        raise ValueError(f"No frames in file: {bvh_path}")
    
    # statistic: mean, std, min, max → shape = [4, C]
    stats = np.vstack([
        np.mean(frames, axis=0),
        np.std(frames, axis=0),
        np.min(frames, axis=0),
        np.max(frames, axis=0),
    ])  # shape = [4, C]
    
    return stats.flatten()  # → shape = [C * 4]

# read the label
def load_label_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    label_dict = load_label_dict(LABELS_PATH)

    data = []
    for filename in os.listdir(SEGMENT_DIR):
        if not filename.endswith('.bvh'):
            continue
        try:
            motion_id = filename.split('_')[0]  # e.g., 05_10_0.bvh → 05_10
            if motion_id not in label_dict:
                print(f"❌ No label for: {motion_id}")
                continue
            label = label_dict[motion_id]['category']
            bvh_path = os.path.join(SEGMENT_DIR, filename)
            feature_vector = extract_statistics_from_segment(bvh_path)
            row = [filename] + feature_vector.tolist() + [label]
            data.append(row)
            print(f"✅ Processed: {filename}")
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")
    
    # creat DataFrame and save
    if data:
        feature_dim = len(data[0]) - 2  # exclude filename and label
        columns = ['motion_id'] + [f'feature_{i+1}' for i in range(feature_dim)] + ['Label']
        df = pd.DataFrame(data, columns=columns)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Saved feature file to: {OUTPUT_FILE}")
    else:
        print("❌ No data processed.")

if __name__ == '__main__':
    main()
