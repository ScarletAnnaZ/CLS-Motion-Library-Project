import os
import json
import numpy as np
import pandas as pd
from bvh import Bvh
import joblib

# ==== 路径配置 ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AKOB_BVH_FILE = os.path.join(BASE_DIR, 'input_AKOB', '1stmay', 'Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh')
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'knn_model.pkl')
CHANNEL_JSON = os.path.join(BASE_DIR, 'output', 'features', 'extract_joint_channels.json')
OUTPUT_CSV = os.path.join(BASE_DIR, 'output', 'akob_label_list.csv')

# ==== 加载 BVH ====
def read_bvh(filepath):
    with open(filepath, 'r') as f:
        bvh = Bvh(f.read())
    return bvh

# ==== 特征提取：按照 joint_channels 顺序提取每帧 96 通道值 ====
def extract_framewise_features(bvh, joint_channels):
    all_frames = []

    for i in range(len(bvh.frames)):
        frame_data = []
        for jc in joint_channels:
            joint, ch = jc.split("_")
            try:
                val = float(bvh.frame_joint_channel(i, joint, ch))
            except:
                val = 0.0  # 如果该 joint 或通道不存在，则设为 0
            frame_data.append(val)
        all_frames.append(frame_data)

    return np.array(all_frames)  # shape = (num_frames, 96)

# ==== 主流程 ====
def main():
    print(f"📂 Loading BVH file: {AKOB_BVH_FILE}")
    bvh = read_bvh(AKOB_BVH_FILE)

    print("📂 Loading joint-channel order...")
    with open(CHANNEL_JSON) as f:
        joint_channels = json.load(f)

    print(f"✅ Extracting framewise features (channels: {len(joint_channels)})...")
    frame_features = extract_framewise_features(bvh, joint_channels)
    print(f"✅ Extracted frame features: {frame_features.shape}")

    print("✅ Loading trained KNN model...")
    knn = joblib.load(MODEL_PATH)

    print("✍️ Predicting...")
    label_list = []
    for i in range(len(frame_features)):
        feature_vec = frame_features[i].reshape(1, -1)
        label = knn.predict(feature_vec)[0]
        label_list.append((i, label))  # 或改为 time = i / FPS

    df = pd.DataFrame(label_list, columns=["Frame", "Predicted Label"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Prediction saved to: {OUTPUT_CSV}")

# ===============================
#  ！！ Aggregating every 600 frames--6s
# ===============================
    print("\n🔄 Aggregating every 600 frames...")

    SEGMENT_SIZE = 600  # Each section has 600 frames
    FPS = 100  # AKOB frame rate

    segment_labels = []
    num_segments = len(label_list) // SEGMENT_SIZE

    for s in range(num_segments):
        segment = label_list[s * SEGMENT_SIZE : (s + 1) * SEGMENT_SIZE]
        labels = [lbl for _, lbl in segment]
        dominant = max(set(labels), key=labels.count)  # voting
        start_sec = s * SEGMENT_SIZE / FPS
        end_sec = (s + 1) * SEGMENT_SIZE / FPS
        segment_labels.append((f"{start_sec:.2f}-{end_sec:.2f} sec", dominant))
        print(f"🟢 {start_sec:.2f}-{end_sec:.2f}s → {dominant}")

    # csv
    segment_df = pd.DataFrame(segment_labels, columns=["Time Segment", "Dominant Label"])
    segment_csv = os.path.join(BASE_DIR, 'output', 'akob_segment_labels.csv')
    segment_df.to_csv(segment_csv, index=False)
    print(f"\n✅ Segment-level label list saved to {segment_csv}")


if __name__ == "__main__":
    main()
