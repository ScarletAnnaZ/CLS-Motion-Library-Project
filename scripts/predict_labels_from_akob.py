import os
import numpy as np
import pandas as pd
from bvh import Bvh
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AKOB_BVH_FILE = os.path.join(BASE_DIR, 'input_AKOB', '1stmay', 'Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh')  # 修改为你具体文件名
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'knn_model.pkl')
OUTPUT_CSV = os.path.join(BASE_DIR, 'output', 'akob_label_list.csv')
CHANNEL_JSON = os.path.join(BASE_DIR, 'output', 'features', 'extract_joint_channels.json')
with open(CHANNEL_JSON) as f:
    joint_channels = json.load(f)


# read bvh
def read_bvh(filepath):
    with open(filepath, 'r') as f:
        bvh = Bvh(f.read())
    return bvh

#extract features

def extract_framewise_features(bvh, joint_channels):
    all_frames = []

    for i in range(len(bvh.frames)):
        frame_data = []
        for jc in joint_channels:
            joint, ch = jc.split("_")
            try:
                val = float(bvh.frame_joint_channel(i, joint, ch))
            except:
                val = 0.0
            frame_data.append(val)
        all_frames.append(frame_data)

    return np.array(all_frames)  # shape = (num_frames, 96)


# process
def main():
    print(f"📂 Loading BVH file: {AKOB_BVH_FILE}")
    bvh = read_bvh(AKOB_BVH_FILE)
    frames = np.array(bvh.frames, dtype=float)
    total_frames = len(frames)

    print(f"✅ Loaded {total_frames} frames (100FPS)")
    
    WINDOW = 600  # Each section has 600 frames -- 6s
    label_list = []
    knn = joblib.load(MODEL_PATH)

    for i in range(0, total_frames - WINDOW + 1, WINDOW):
        segment = frames[i:i+WINDOW]
        features = extract_summary_features(segment).reshape(1, -1)  # shape (1, N)
        label = knn.predict(features)[0]
        start_sec = i / 100  # 100FPS，转换成秒
        end_sec = (i + WINDOW) / 100
        label_list.append((f"{start_sec:.2f}-{end_sec:.2f} sec", label))
        print(f"🟢 {start_sec:.2f}-{end_sec:.2f}s → {label}")

    # CSV
    df = pd.DataFrame(label_list, columns=["Time Segment", "Predicted Label"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Label list saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
