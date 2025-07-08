import os
import json
import re

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENT_DIR = os.path.join(BASE_DIR, 'output2', 'segments_600')
BASE_LABEL_PATH = os.path.join(BASE_DIR, 'output2', 'dance_labels.json')
OUTPUT_LABEL_PATH = os.path.join(BASE_DIR, 'output2', 'segment_labels_600.json')

# read basic label（original motion's label）
with open(BASE_LABEL_PATH, 'r', encoding='utf-8') as f:
    base_labels = json.load(f)

# creat lables segment to segment 
segment_labels = {}
unmatched = []

for fname in os.listdir(SEGMENT_DIR):
    if not fname.endswith('.bvh'):
        continue

    match = re.match(r'(.+)_seg(\d+)', fname)
    if not match:
        continue

    motion_id, seg_idx = match.groups()  # notw: no .bvh

    if motion_id in base_labels:
        segment_labels[fname] = {
            "category": base_labels[motion_id]["category"],
            "description": base_labels[motion_id]["description"]
        }
    else:
        unmatched.append(fname)

# save as JSON
with open(OUTPUT_LABEL_PATH, 'w', encoding='utf-8') as f:
    json.dump(segment_labels, f, indent=4, ensure_ascii=False)

# output
print(f"✅ Segment labels saved to: {OUTPUT_LABEL_PATH}")
print(f"🔍 Matched segments: {len(segment_labels)}")
print(f"⚠️ Unmatched segments: {len(unmatched)}")
if unmatched:
    print("Examples of unmatched:", unmatched[:5])
