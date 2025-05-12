# label_to_action_mapping.py

LABEL_TO_ACTION = {
    "salsa dance ": "salsa dance",
     "follow": "move_forward",
    "dodge": "move_sideways",
    "mirror": "mirror_dancer",
    "highlight": "increase_intensity",
    "idle": "reset_position",
    "turn": "rotate_agent",
    "jump": "jump_response"
}


def get_agent_action(label: str) -> str:
    """
    Given a predicted motion label, return the corresponding agent action.
    """
    return LABEL_TO_ACTION.get(label, "unknown_action")

if __name__ == "__main__":
    test_labels = ["follow", "mirror", "highlight", "idle", "dance"]
    for label in test_labels:
        action = get_agent_action(label)
        print(f"Label: {label} → Agent Action: {action}")


