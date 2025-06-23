import os
import json
import pandas as pd
import random
from label_to_action import get_agent_action  


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTED_LABEL_FILE = os.path.join(BASE_DIR, 'output2', 'predicted_segments.csv')
STANDARD_LABEL_FILE = os.path.join(BASE_DIR, 'output2', 'segment_labels_600.json')
MOTION_DATA_DIR = os.path.join(BASE_DIR, 'output2','segments_600')  # motion_id like 93_05 → data/93/93_05.bvh
OUTPUT_FILE = os.path.join(BASE_DIR, 'output2', 'predicted_segment_responses.csv')

STRATEGY = "mirror"  #change the strate

#加载 motion 标签库 
with open(STANDARD_LABEL_FILE, 'r', encoding='utf-8') as f:
    label_map = json.load(f)

# category → [motion_id]
label_to_motion_ids = {}
for motion_id, info in label_map.items():
    label = info.get("category", "").strip()
    label_to_motion_ids.setdefault(label, []).append(motion_id)

# mapping the label, select motion 
def sample_motion_by_strategy(predicted_label: str) -> str:
    response_label = get_agent_action(predicted_label.strip(), strategy=STRATEGY)
    candidates = label_to_motion_ids.get(response_label, [])
    if not candidates:
        return "no_motion_found"
    return f"{random.choice(candidates)}"


#
def main():
    df = pd.read_csv(PREDICTED_LABEL_FILE)

    # 得到 Response Label 列
    df["Response Label"] = df["Predicted Label"].astype(str).apply(lambda x: get_agent_action(x.strip(), strategy=STRATEGY))

    # 根据 Response Label 随机选 motion
    df["Selected Motion"] = df["Response Label"].apply(sample_motion_by_strategy)

    # 保存结果
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Strategy='{STRATEGY}' responses saved to: {OUTPUT_FILE}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
