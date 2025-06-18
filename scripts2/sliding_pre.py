import os
import json
import numpy as np
import joblib
from bvh import Bvh
import pandas as pd

# ==== 配置路径 ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_BVH = os.path.join(BASE_DIR, 'input_AKOB', '1stmay', 'Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh')
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'knn_seg120.pkl')
CHANNEL_LIST_PATH = os.path.join(BASE_DIR, "output", "features", "extract_joint_channels.json")
OUTPUT_CSV = os.path.join(BASE_DIR, 'output2', 'predicted_sliding_segments.csv')

SEGMENT_LENGTH = 120
STEP_SIZE = 60  # 每次滑动多少帧

def read_bvh(filepath):
    with open(filepath, "r") as f:
        return Bvh(f.read())

def get_channel_indices(bvh_data, required_channels):
    available_channels = []
    for joint in bvh_data.get_joints():
        name = joint.name
        try:
            for ch in bvh_data.joint_channels(name):
                available_channels.append(f"{name}_{ch}")
        except:
            continue
    return [available_channels.index(ch) if ch in available_channels else None for ch in required_channels]

def extract_sliding_features(frames, indices, segment_len, step):
    segment_features = []
    segments_meta = []

    for start in range(0, len(frames) - segment_len + 1, step):
        segment = np.array(frames[start:start+segment_len], dtype=float)
        features = []
        for idx in indices:
            if idx is not None:
                values = segment[:, idx]
                features.extend([np.mean(values), np.std(values), np.min(values), np.max(values)])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        segment_features.append(features)
        segments_meta.append((start, start + segment_len - 1))

    return segment_features, segments_meta

def main():
    bvh_data = read_bvh(INPUT_BVH)
    frames = bvh_data.frames

    with open(CHANNEL_LIST_PATH, "r") as f:
        required_channels = json.load(f)

    indices = get_channel_indices(bvh_data, required_channels)
    features, segments_meta = extract_sliding_features(frames, indices, SEGMENT_LENGTH, STEP_SIZE)
    
    if not features:
        print("❌ No valid segments found.")
        return

    X = np.array(features)

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)

    # 保存结果为 CSV
    df = pd.DataFrame({
        "segment_id": [f"seg{i}" for i in range(len(preds))],
        "start_frame": [meta[0] for meta in segments_meta],
        "end_frame": [meta[1] for meta in segments_meta],
        "predicted_label": preds
    })

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved sliding prediction to {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
