import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(BASE_DIR, 'output', 'motion_labels.json')

# read the json file
with open(json_path, 'r', encoding='utf-8') as f:
    motion_labels = json.load(f)

# show the top 5  motion_id and its label
for i, (motion_id, labels) in enumerate(motion_labels.items()):
    print(f"{motion_id}: Short Label = {labels['short_label']}, Long Label = {labels['long_label']}")
    if i >= 4:  
        break
