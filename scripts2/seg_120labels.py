import os
import json
import re

# ==== 路径设置 ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENT_DIR = os.path.join(BASE_DIR, 'output2', 'segments_600')
BASE_LABEL_PATH = os.path.join(BASE_DIR, 'output2', 'dance_related_labels.json')
OUTPUT_LABEL_PATH = os.path.join(BASE_DIR, 'output2', 'segment_labels_600.json')

# 读取基础 label（原始 motion_id 的标签）
with open(BASE_LABEL_PATH, 'r', encoding='utf-8') as f:
    base_labels = json.load(f)

# 构建 segment 对应的标签
segment_labels = {}

for fname in os.listdir(SEGMENT_DIR):
    if not fname.endswith('.bvh'):
        continue

    match = re.match(r'(.+)_seg(\d+)\.bvh', fname)
    if not match:
        continue

    motion_id, seg_idx = match.groups()
    if motion_id in base_labels:
        segment_labels[fname] = {
            "category": base_labels[motion_id]["category"],
            "description": base_labels[motion_id]["description"]
        }

# 保存 JSON
with open(OUTPUT_LABEL_PATH, 'w', encoding='utf-8') as f:
    json.dump(segment_labels, f, indent=4, ensure_ascii=False)

print(f"✅ Segment labels saved to: {OUTPUT_LABEL_PATH}")
