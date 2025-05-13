# label_to_action_mapping.py

LABEL_TO_ACTION = {
    "salsa dance": "salsa dance",
    "snake (human subject)": "snake (human subject)",
    "nursery rhyme - Cock Robin": "nursery rhyme - Cock Robin",
    "elephant (human subject)": "elephant (human subject)",
    "walking, running, kicking, punching, knee kicking, and stretching":"walking, running, kicking, punching, knee kicking, and stretching"
}
def get_agent_action(label: str) -> str:
    """
    Given a predicted motion label, return the corresponding agent action.
    """
    return LABEL_TO_ACTION.get(label, "unknown_action")


if __name__ == "__main__":
    test_labels = ["elephant (human subject)", "salsa dance", "highlight", "nursery rhyme - Cock Robin", "snake (human subject)"]
    for label in test_labels:
        action = get_agent_action(label)
        print(f"Label: {label} → Agent Action: {action}")
