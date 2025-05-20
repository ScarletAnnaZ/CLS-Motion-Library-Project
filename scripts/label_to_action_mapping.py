# label_to_action_mapping.py

LABEL_TO_ACTION = {
    "salsa": "salsa",
    "various everyday behaviors": "various everyday behaviors",
    "recreation, nursery rhymes, animal behaviors (pantomime - human subject": "recreation, nursery rhymes, animal behaviors (pantomime - human subject",
    "elephant (human subject)": "elephant (human subject)",
    "walking, running, kicking, punching, knee kicking, and stretching":"walking, running, kicking, punching, knee kicking, and stretching"
}
def get_agent_action(label: str) -> str:
    """
    Given a predicted motion label, return the corresponding agent action.
    """
    return LABEL_TO_ACTION.get(label, "unknown_action")


if __name__ == "__main__":
    test_labels = ["various everyday behaviors", "salsa", "recreation, nursery rhymes, animal behaviors (pantomime - human subject)"]
    for label in test_labels:
        action = get_agent_action(label)
        print(f"Label: {label} → Agent Action: {action}")
