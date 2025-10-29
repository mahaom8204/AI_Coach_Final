import os
import json

def load_roadmap():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "English_Roadmap.json"
    )
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)["English_Learning_Roadmap"]

def flatten_roadmap():
    rm = load_roadmap()
    flat = []
    for level, info in rm.items():
        for key, val in info.items():
            if isinstance(val, dict) and "Description" in val:
                flat.append({
                    "level": level,
                    "title": key,
                    "desc": val["Description"],
                    "examples": val.get("Example", [])
                })
    return flat
