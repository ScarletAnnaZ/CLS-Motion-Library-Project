import os
import json
import pandas as pd
import shutil
import re

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSON = os.path.join(BASE_DIR, 'output', 'standardized_labels.json')
OUTPUT_JSON = os.path.join(BASE_DIR, 'output2', 'dance_related_labels.json')
MOTION_DATA_DIR = os.path.join(BASE_DIR, 'data') 
DANCE_LIBRARY_DIR = os.path.join(BASE_DIR, 'data_clean', 'dance_motion_library')
os.makedirs(DANCE_LIBRARY_DIR, exist_ok=True)


# 舞蹈关键词白名单
DANCE_KEYWORDS = [
    'dance', 'salsa', 'tango', 'ballet', 'waltz','Dance','hippo','yoga',
    'chacha', 'rumba', 'jazz', 'jitterbug', 'boogie', 'robot', 'snake','recreation',
    'freestyle', 'club', 'disco', 'nursery rhyme - Cock Robin', 
    'various everyday behaviors', 
    'Varying Weird Walks','gymnastics'
    
]
dance_patterns = [re.compile(rf'\b{re.escape(kw)}\b') for kw in DANCE_KEYWORDS]

#def is_dance_related(text):
 #   text = text.lower()
  #  return any(keyword in text for keyword in DANCE_KEYWORDS)

def is_dance_related(text):
    text = text.lower()
    for pattern in dance_patterns:
        if pattern.search(text):
            # print(f"✅ Keyword matched: '{pattern.pattern}' in '{text}'") # to check how to select the data
            return True
    return False

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




# creat the new library
def copy_dance_motions(filtered_json, source_root, target_dir):
    with open(filtered_json, 'r', encoding='utf-8') as f:
        filtered = json.load(f)

    os.makedirs(target_dir, exist_ok=True)
    count = 0

    for motion_id in filtered.keys():
        if not motion_id.endswith('.bvh'):
            motion_id += '.bvh'

        prefix = motion_id.split('_')[0]  # e.g., '05' from '05_10.bvh'
        source_path = os.path.join(source_root, prefix, motion_id)

        if not os.path.exists(source_path):
            print(f"⚠️ File not found: {source_path}")
            continue

        shutil.copy(source_path, os.path.join(target_dir, motion_id))
        count += 1

    print(f"✅ Copied {count} motion files to {target_dir}")


if __name__ == "__main__":
    filter_dance_labels(INPUT_JSON, OUTPUT_JSON)
    copy_dance_motions(OUTPUT_JSON, MOTION_DATA_DIR, DANCE_LIBRARY_DIR)

