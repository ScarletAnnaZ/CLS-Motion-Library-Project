import os
import json
import pandas as pd

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSON = os.path.join(BASE_DIR, 'output', 'standardized_labels.json')
OUTPUT_JSON = os.path.join(BASE_DIR, 'output2', 'dance_related_labels.json')

# 舞蹈关键词白名单
DANCE_KEYWORDS = [
    'dance', 'salsa', 'tango', 'ballet', 'waltz',
    'chacha', 'rumba', 'jazz', 'jitterbug', 'boogie', 'robot',
    'freestyle', 'club', 'disco', 'nursery rhyme - Cock Robin','various everyday behaviors','Varying Weird Walks',
    'recreation, nursery rhymes, animal behaviors (pantomime - human subject',
    
]

def is_dance_related(text):
    text = text.lower()
    return any(keyword in text for keyword in DANCE_KEYWORDS)

def filter_dance_labels(input_path, output_json):
    with open(input_path, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)

    filtered = {}
    for motion_id, info in labels_dict.items():
        desc = info.get("description", "")
        cat = info.get("category", "")
        combined = f"{desc} {cat}"
        if is_dance_related(combined):
            filtered[motion_id] = info

    print(f"Found {len(filtered)} dance-related motions.")

    # 保存为 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=4, ensure_ascii=False)


    print(f"✅ Saved to {output_json}")

if __name__ == "__main__":
    filter_dance_labels(INPUT_JSON, OUTPUT_JSON)
