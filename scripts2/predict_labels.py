import os
import json
import numpy as np
import joblib
from bvh import Bvh
import pandas as pd

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#INPUT_BVH = os.path.join(BASE_DIR, 'input_AKOB', '1stmay', 'Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh')
INPUT_BVH = os.path.join(BASE_DIR, 'data', '13', '13_17.bvh')  # replace input
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'rf_600model.pkl')  # can be replaced by the differ models
CHANNEL_LIST_PATH = os.path.join(BASE_DIR, "output", "features", "extract_joint_channels.json")
OUTPUT_CSV = os.path.join(BASE_DIR, 'output2', 'predicted_segments.csv')
SEGMENT_LENGTH = 600

def read_bvh(filepath):
    with open(filepath, "r") as f:
        bvh_data = Bvh(f.read())
    return bvh_data

# Match desired channels to indices in the BVH file
def get_channel_indices(bvh_data, required_channels):
    available_channels = []
    for joint in bvh_data.get_joints():
        name = joint.name
        try:
            for ch in bvh_data.joint_channels(name):
                available_channels.append(f"{name}_{ch}")
        except:
            continue
    indices = []
    for ch in required_channels:
        if ch in available_channels:
            idx = available_channels.index(ch)
            indices.append(int(idx))  #  # Force cast to int
        else:
            indices.append(None)
    return indices

# Extract segment-level statistical features (mean, std, min, max) for selected channels
def extract_segment_features(frames, indices):
    segment_features = []
    for i in range(0, len(frames) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):
        segment = np.array(frames[i:i+SEGMENT_LENGTH], dtype=float)
        selected = []
        for idx in indices:
            if not isinstance(idx, int):
                selected.extend([0.0, 0.0, 0.0, 0.0])
                continue
            if idx < 0 or idx >= segment.shape[1]:
            
                selected.extend([0.0, 0.0, 0.0, 0.0])
                continue
            try:
                values = segment[:, idx]
                selected.extend([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values)
                ])
            except Exception as e:
                print(f"⚠️ Warning: error at idx {idx}: {e}")
                selected.extend([0.0, 0.0, 0.0, 0.0])
        segment_features.append(selected)
    return segment_features


# Classify all motion segments from a BVH file
def classify_segment(input_bvh_path):
    """
    Input a BVH file path and return a list of predicted labels for each 600-frame segment
    """
    bvh_data = read_bvh(input_bvh_path)
    frames = bvh_data.frames

    with open(CHANNEL_LIST_PATH, "r") as f:
        required_channels = json.load(f)

    indices = get_channel_indices(bvh_data, required_channels)
    segment_features = extract_segment_features(frames, indices)

    if not segment_features:
        print("❌ The input frames are less than SEGMENT_LENGTH. No valid segments found.")
        return []

    model = joblib.load(MODEL_PATH)
    X = np.array(segment_features)
    preds = model.predict(X)

    return preds.tolist()

def main():
    # load bvh
    bvh_data = read_bvh(INPUT_BVH)
    frames = bvh_data.frames

    # load channel list
    with open(CHANNEL_LIST_PATH, "r") as f:
        required_channels = json.load(f)

    indices = get_channel_indices(bvh_data, required_channels)

    # extract features
    segment_features = extract_segment_features(frames, indices)
    if not segment_features:
        print("❌ The input frames are less than SEGMENT_LENGTH. No valid segments found.")
        return

    # load model
    model = joblib.load(MODEL_PATH)

    # predict the labels
    X = np.array(segment_features)
    preds = model.predict(X)

    print("✅ Predicted Labels:")
    for i, label in enumerate(preds):
        print(f"Segment {i}: {label}")
    
     # save the result as CSV
    df = pd.DataFrame({
    "Time Segment": [f"segment_{i}" for i in range(len(preds))],
    "Predicted Label": preds
})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Prediction results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
    
'''
if __name__ == "__main__":
    preds = classify_segment(INPUT_BVH)
    print("✅ Predicted Labels:")
    for i, label in enumerate(preds):
        print(f"Segment {i}: {label}")
    
    # Save the prediction results as CSV
    df = pd.DataFrame({
        "Time Segment": [f"segment_{i}" for i in range(len(preds))],
        "Predicted Label": preds
    })
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Prediction results saved to: {OUTPUT_CSV}")
'''

