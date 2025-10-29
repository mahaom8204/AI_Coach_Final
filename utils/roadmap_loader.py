# utils/roadmap_loader.py
import os
import json
import pandas as pd

def load_roadmap_dict():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "English_Roadmap.json"
    )
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)["English_Learning_Roadmap"]

def flatten_roadmap():
    rm = load_roadmap_dict()
    topics = []
    for level_name, items in rm.items():
        for key, value in items.items():
            # keys like "1. Fundamentals": {Description: ..., Example: [...]}
            if isinstance(value, dict) and "Description" in value:
                topics.append({
                    "topic": key.replace("1.", "").replace("2.", "").strip(),
                    "roadmap_level": level_name,
                    "description": value["Description"],
                    "examples": value.get("Example", [])
                })
    return pd.DataFrame(topics)
