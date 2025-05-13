import os
import pandas as pd
from label_to_action_mapping import get_agent_action

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_FILE = os.path.join(BASE_DIR, 'output', 'akob_segment_labels.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'output', 'akob_agent_responses.csv')

def main():
    # 读取预测标签 CSV
    df = pd.read_csv(LABEL_FILE)

    # 映射标签为 agent 动作
    df['Agent Action'] = df['Dominant Label'].apply(get_agent_action)

    # 保存为新文件
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Agent action sequence saved to {OUTPUT_FILE}")

    # 可选：打印部分结果
    print("Agent action sequence:")
    print(df.to_string(index=False))
    

if __name__ == "__main__":
    main()
   