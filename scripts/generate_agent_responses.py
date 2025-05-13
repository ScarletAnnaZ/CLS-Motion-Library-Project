import json
import os
import random
import pandas as pd
from label_to_action_mapping import get_agent_action

# ─────────────── 配置 ───────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_FILE      = os.path.join(BASE_DIR, 'output', 'akob_segment_labels.csv')
JSON_LABEL_FILE = os.path.join(BASE_DIR, 'output','standardized_labels.json')
BVH_DIR         = os.path.join(BASE_DIR, 'data')  
OUTPUT_FILE     = os.path.join(BASE_DIR, 'output', 'akob_segment_responses.csv')

random.seed(42)

with open(JSON_LABEL_FILE, 'r', encoding='utf-8') as f:
    std_labels = json.load(f)

    library_index = {}
for motion_id, info in std_labels.items():
    desc = info['description']
    library_index.setdefault(desc, []).append(motion_id)


# Randomly select a motion from the description
def sample_motion(description_label: str) -> str:
    """
    给定一个 description（比如 "playground - climb"），
    随机挑一个 motion_id 并加上 .bvh 后缀。
    """
    key = description_label.strip()
    candidates = library_index.get(key, [])
    if not candidates:
        return 'no_motion_found'
    chosen = random.choice(candidates)
    return f"{chosen}.bvh"

def main():
    # 3.1 读取预测的 segment labels
    df = pd.read_csv(LABEL_FILE)

    # 3.2 label → response_label（即 description）  
    # 注意：get_agent_action 里要返回和 standardized_labels.json 中 description 完全一致的字符串
    df['Response Label'] = df['Dominant Label'].astype(str).apply(lambda x: get_agent_action(x.strip()))

    # 3.3 response_label → selected motion
    df['Selected Motion'] = df['Response Label'].apply(sample_motion)

    # 3.4 保存结果
    #df.to_csv(OUTPUT_FILE, index=False)
    #print(f"✅ 已生成 agent response 文件：{OUTPUT_FILE}")
    # 3.4 打印结果（不写 CSV）
    print(f"{'Time Segment':<15} | {'Dominant Label':<40} | {'Response Label':<30} | {'Selected Motion'}")
    print("-"*15, "+", "-"*40, "+", "-"*30, "+", "-"*15)
    for _, row in df.iterrows():
        ts = row['Time Segment']
        dl = row['Dominant Label']
        rl = row['Response Label']
        sm = row['Selected Motion']
        print(f"{ts:<15} | {dl:<40} | {rl:<30} | {sm}")


if __name__ == "__main__":
    main()