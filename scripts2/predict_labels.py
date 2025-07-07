import os
import json
import numpy as np
import joblib
from bvh import Bvh
import pandas as pd

# ==== 配置路径 ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#INPUT_BVH = os.path.join(BASE_DIR, 'input_AKOB', '1stmay', 'Take 2020-05-01 11.26.00_FB_mirror,follow,drones_follow.bvh')
INPUT_BVH = os.path.join(BASE_DIR, 'data', '13', '13_17.bvh')  # replace input
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'rf_600model.pkl')  # 替换为训练好的模型路径
CHANNEL_LIST_PATH = os.path.join(BASE_DIR, "output", "features", "extract_joint_channels.json")
OUTPUT_CSV = os.path.join(BASE_DIR, 'output2', 'predicted_segments.csv')
SEGMENT_LENGTH = 600

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
            indices.append(int(idx))  # ✅ 强制转换为 int 类型！
        else:
            indices.append(None)
    return indices


def extract_segment_features(frames, indices):
    segment_features = []
    for i in range(0, len(frames) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):
        segment = np.array(frames[i:i+SEGMENT_LENGTH], dtype=float)
        selected = []
        for idx in indices:
            if not isinstance(idx, int):
                # 如果是 float、None 或其他非法类型，跳过
                selected.extend([0.0, 0.0, 0.0, 0.0])
                continue
            if idx < 0 or idx >= segment.shape[1]:
                # 越界索引也跳过
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



def classify_segment(input_bvh_path):
    """
    输入一个 BVH 文件路径，返回每个 600 帧 segment 的预测标签列表。
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
'''
if __name__ == "__main__":
    preds = classify_segment(INPUT_BVH)
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
'''

