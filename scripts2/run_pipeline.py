import os
import pandas as pd
import joblib
import numpy as np
from bvh import Bvh
import json
import random

# Path setting
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_BVH = os.path.join(BASE_DIR, 'data', '13', '13_12.bvh') # change the input data
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'rf_600model.pkl') # change the different model-from 'output/models'
CHANNEL_LIST_PATH = os.path.join(BASE_DIR, "output", "features", "extract_joint_channels.json") # the file that records the channels used in the feature extractions
SEGMENT_LENGTH = 600 # the frame size used for trainned dataset and output data
PREDICTED_LABEL_FILE = os.path.join(BASE_DIR, 'output2', 'predicted_segments.csv') # file -- predicted segments' labels
STANDARD_LABEL_FILE = os.path.join(BASE_DIR, 'output2', 'segment_labels_600.json')
SEGMENT_DIR = os.path.join(BASE_DIR, 'output2','segments_600') # motion library path
RESPONSE_FILE = os.path.join(BASE_DIR, 'output2', 'predicted_segment_responses.csv') # file -- predicted segments' labels + response label + selected motion id
MERGED_OUTPUT_BVH = os.path.join(BASE_DIR, 'output2', 'merged_response1.bvh') # generate final response motion
STRATEGY = "mirror" # the response strategy used


# Module function
def read_bvh(filepath):
    with open(filepath, "r") as f:
        bvh_data = Bvh(f.read())
    return bvh_data

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
            indices.append(int(idx))
        else:
            indices.append(None)
    return indices

def extract_segment_features(frames, indices):
    segment_features = []
    for i in range(0, len(frames) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):
        segment = np.array(frames[i:i+SEGMENT_LENGTH], dtype=float)
        selected = []
        for idx in indices:
            if not isinstance(idx, int) or idx < 0 or idx >= segment.shape[1]:
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
            except:
                selected.extend([0.0, 0.0, 0.0, 0.0])
        segment_features.append(selected)
    return segment_features

def step1_predict_labels():
    print("1️⃣ Step 1: Predicting motion labels...")
    bvh_data = read_bvh(INPUT_BVH)
    frames = bvh_data.frames
    with open(CHANNEL_LIST_PATH, "r") as f:
        required_channels = json.load(f)
    indices = get_channel_indices(bvh_data, required_channels)
    segment_features = extract_segment_features(frames, indices)
    if not segment_features:
        print("❌ No valid segments found.")
        return
    model = joblib.load(MODEL_PATH)
    preds = model.predict(np.array(segment_features))
    df = pd.DataFrame({
        "Time Segment": [f"segment_{i}" for i in range(len(preds))],
        "Predicted Label": preds
    })
    df.to_csv(PREDICTED_LABEL_FILE, index=False)
    print(f"✅ Saved predicted labels to: {PREDICTED_LABEL_FILE}")

def get_agent_action(label: str, strategy: str = "mirror") -> str:
    label = label.strip()
    if strategy == "mirror":
        return label
    else:
        return "invalid_strategy"

def sample_motion_by_strategy(predicted_label: str, label_map: dict) -> str:
    response_label = get_agent_action(predicted_label.strip(), strategy=STRATEGY)
    candidates = label_map.get(response_label, [])
    return f"{random.choice(candidates)}" if candidates else "no_motion_found"

def step2_generate_responses():
    print("2️⃣ Step 2: Generating agent responses...")
    with open(STANDARD_LABEL_FILE, 'r', encoding='utf-8') as f:
        full_label_map = json.load(f)
    label_to_motion_ids = {}
    for motion_id, info in full_label_map.items():
        label = info.get("category", "").strip()
        label_to_motion_ids.setdefault(label, []).append(motion_id)

    df = pd.read_csv(PREDICTED_LABEL_FILE)
    df["Response Label"] = df["Predicted Label"].astype(str).apply(lambda x: get_agent_action(x.strip(), strategy=STRATEGY))
    df["Selected Motion"] = df["Response Label"].apply(lambda x: sample_motion_by_strategy(x, label_to_motion_ids))
    df.to_csv(RESPONSE_FILE, index=False)
    print(f"✅ Saved response mapping to: {RESPONSE_FILE}")

def step3_merge_responses():
    print("3️⃣ Step 3: Merging response segments into one .bvh file...")
    df = pd.read_csv(RESPONSE_FILE)
    file_list = df["Selected Motion"].tolist()
    all_frames = []
    frame_time = None
    skeleton_text = None

    for idx, fname in enumerate(file_list):
        if fname == "no_motion_found":
            continue
        motion_path = os.path.join(SEGMENT_DIR, fname)
        if not os.path.exists(motion_path):
            print(f"⚠️ Skipping missing file: {motion_path}")
            continue
        with open(motion_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        bvh = Bvh(raw)
        if idx == 0:
            skeleton_text = raw.split("MOTION")[0]
            frame_time = bvh.frame_time
        all_frames.extend(bvh.frames)

    if all_frames:
        with open(MERGED_OUTPUT_BVH, 'w', encoding='utf-8') as fout:
            fout.write(skeleton_text)
            fout.write("MOTION\n")
            fout.write(f"Frames: {len(all_frames)}\n")
            fout.write(f"Frame Time: {frame_time:.6f}\n")
            for frame in all_frames:
                fout.write(" ".join(frame) + "\n")
        print(f"✅ Merged BVH saved to: {MERGED_OUTPUT_BVH}")
    else:
        print("❌ No valid motions to merge.")

# print the predicted labels and selected response motion_id
print("\n🔍 Preview of label → motion mapping:")
df = pd.read_csv(RESPONSE_FILE)
print(df[["Time Segment", "Predicted Label", "Selected Motion"]])


# === 主程序入口 ===
if __name__ == "__main__":
    step1_predict_labels()
    step2_generate_responses()
    step3_merge_responses()
