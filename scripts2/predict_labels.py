import os
import json
import numpy as np
import joblib
from bvh import Bvh
import pandas as pd

# ==== 配置路径 ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#INPUT_BVH = os.path.join(BASE_DIR, 'input_AKOB', '1stmay', 'Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh')
INPUT_BVH = os.path.join(BASE_DIR, 'data', '93', '93_02.bvh')  # 替换为实际输入
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'rf_600model.pkl')  # 替换为训练好的模型路径
CHANNEL_LIST_PATH = os.path.join(BASE_DIR, "output", "features", "extract_joint_channels.json")
OUTPUT_CSV = os.path.join(BASE_DIR, 'output2', 'predicted_segments.csv')
SEGMENT_LENGTH = 600

def read_bvh(filepath):
    with open(filepath, "r") as f:
        bvh_data = Bvh(f.read())
    return bvh_data

def get_channel_indices(bvh_data, required_channels):
    """
    查找所需通道在当前 bvh 文件中的索引，如果缺失则填充为 None
    """
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
            indices.append(available_channels.index(ch))
        else:
            indices.append(None)  # 缺失通道
    return indices

def extract_segment_features(frames, indices):
    """
    提取 segment 特征（mean, std, min, max）忽略缺失通道
    """
    segment_features = []
    for i in range(0, len(frames) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):
        segment = np.array(frames[i:i+SEGMENT_LENGTH], dtype=float)
        selected = []
        for idx in indices:
            if idx is not None:
                values = segment[:, idx]
                selected.extend([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values)
                ])
            else:
                selected.extend([0.0, 0.0, 0.0, 0.0])  # 缺失通道用0填充
        segment_features.append(selected)
    return segment_features

def main():
    # 加载 bvh
    bvh_data = read_bvh(INPUT_BVH)
    frames = bvh_data.frames

    # 加载通道列表
    with open(CHANNEL_LIST_PATH, "r") as f:
        required_channels = json.load(f)

    indices = get_channel_indices(bvh_data, required_channels)

    # 提取特征
    segment_features = extract_segment_features(frames, indices)
    if not segment_features:
        print("❌ The input frames are less than SEGMENT_LENGTH. No valid segments found.")
        return

    # 加载模型
    model = joblib.load(MODEL_PATH)

    # 预测
    X = np.array(segment_features)
    preds = model.predict(X)

    print("✅ Predicted Labels:")
    for i, label in enumerate(preds):
        print(f"Segment {i}: {label}")
    
     # 保存预测结果为 CSV
    df = pd.DataFrame({
    "Time Segment": [f"segment_{i}" for i in range(len(preds))],
    "Predicted Label": preds
})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Prediction results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
    # 保存预测结果为 CSV

