import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSON = os.path.join(BASE_DIR, 'output2', 'dance_related_labels.json')
OUTPUT_JSON = os.path.join(BASE_DIR, 'output2', 'dance_labels.json')

CATEGORY_MAPPING = {
    "recreation, nursery rhymes, animal behaviors (pantomime - human subject":"pantomime",
    "various everyday behaviors": "various everyday behaviors",
    "salsa": "salsa", 
    "modern dance, gymnastics": "modern_dance",
    "modern dance": "modern_dance",
    "indian dance": "indian_dance",
    "various everyday behaviors, dance moves": "various everyday behaviors",
    "recreation, nursery rhymes": "nursery rhymes motion",
    "Charleston Dance": "Charleston Dance",
    "Various Style Walks": "general movement",
    "animal behaviors (pantomime - human subject": "animal behavior",
    "General Subject Capture": "general movement",
    "cartwheels; acrobatics; dances":"modern_dance",
    "human interaction - at play, formations (2 subjects - subject A": "interaction",
    "human interaction - at play, formations (2 subjects - subject B": "interaction",
    "human interaction and communication (2 subjects - subject A": "interaction",
    "human interaction and communication (2 subjects - subject B": "interaction",
    "actor everyday activities": "various everyday behaviors",
    "assorted motions":"general movement"
}

# 加载原始文件
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    raw_labels = json.load(f)

# 映射新标签
mapped_labels = {}
for motion_id, info in raw_labels.items():
    original_category = info.get("category", "")
    mapped_category = CATEGORY_MAPPING.get(original_category, "other")
    mapped_labels[motion_id] = {
        "description": info.get("description", ""),
        "category": mapped_category
    }

# 保存结果
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(mapped_labels, f, indent=4, ensure_ascii=False)

print(f"✅ Mapped dance categories saved to: {OUTPUT_JSON}")
