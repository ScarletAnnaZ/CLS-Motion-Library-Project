import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
labels_file = os.path.join(BASE_DIR, 'output','standardized_labels.json')

KEYWORDS = ["dance", "salsa", "nursery rhymes", "running", "Varying Weird Walks", "walking","various everyday behaviors"]

# 读取 JSON 文件
with open(labels_file, 'r', encoding='utf-8') as f:
    labels_data = json.load(f)

# 提取符合条件的 description 标签（避免重复）
filtered_descriptions = set()
for info in labels_data.values():
    category = info.get("category", "").lower()
    description = info.get("description", "")
    if any(keyword in category for keyword in KEYWORDS):
        filtered_descriptions.add(description)

# 打印符合条件的 description 列表
for desc in sorted(filtered_descriptions):
    print(f"- {desc}")
