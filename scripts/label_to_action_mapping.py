# label_to_action_mapping.py

LABEL_TO_ACTION = {
    "salsa": "salsa",
    "salsa dance":"salsa dance",
    "snake (human subject)":"snake (human subject)",
    "nursery rhyme - Cock Robin":"nursery rhyme - Cock Robin",
    "walking, running, kicking, punching, knee kicking, and stretching":"walking, running, kicking, punching, knee kicking, and stretching",
    "elephant (human subject)":"elephant (human subject)",
    "various everyday behaviors": "various everyday behaviors",
    "recreation, nursery rhymes, animal behaviors (pantomime - human subject": "recreation, nursery rhymes, animal behaviors (pantomime - human subject",
    "jump": "jump_response",
    "various everyday behaviors": "various everyday behaviors",
    "Varying Weird Walks":"Varying Weird Walks",
    "recreation, nursery rhymes, animal behaviors (pantomime - human subject":"recreation, nursery rhymes, animal behaviors (pantomime - human subject",
    "salsa":"salsa",
   
}


def get_agent_action(label: str) -> str:
    """
    Given a predicted motion label, return the corresponding agent action.
    """
    return LABEL_TO_ACTION.get(label, "unknown_action")

if __name__ == "__main__":
    test_labels = ["elephant (human subject)", "mirror", "highlight", "idle", "dance"]
    for label in test_labels:
        action = get_agent_action(label)
        print(f"Label: {label} → Agent Action: {action}")


