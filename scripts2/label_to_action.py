import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_PATH = os.path.join(BASE_DIR, "output2", "dance_labels.json")

# Load all unique category labels from the JSON label file
def load_all_labels():
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(set(entry["category"] for entry in data.values()))

# Map a motion label to an agent action based on the selected strategy
def get_agent_action(label: str, strategy: str = "mirror") -> str:
    label = label.strip()
    if strategy == "mirror":
        # In mirror strategy, agent simply reflects the same label as action
        return label
    else:
        return "invalid_strategy"

if __name__ == "__main__":
    labels = load_all_labels()  # ← 动态获取，不再写死
    print("=== MIRROR STRATEGY TEST ===")
    for label in labels:
        action = get_agent_action(label, strategy="mirror")
        print(f"Label: {label:30} → Action: {action}")
