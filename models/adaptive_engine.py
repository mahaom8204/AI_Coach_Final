import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EXPORT_DIR = os.path.join(DATA_DIR, "model")
MODEL_PATH = os.path.join(EXPORT_DIR, "dkt_model.h5")
ROADMAP_PATH = os.path.join(DATA_DIR, "English_Roadmap.json")

# GPU config (safe if no GPU)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# lazy singletons
_model = None
_qid_encoder = None
_MAX_LEN = None
_topics_df = None

def _load_topics_df():
    global _topics_df
    if _topics_df is not None:
        return _topics_df

    with open(ROADMAP_PATH, "r", encoding="utf-8") as f:
        roadmap = json.load(f)["English_Learning_Roadmap"]

    topics = []
    for level_name, items in roadmap.items():
        for key, value in items.items():
            if isinstance(value, dict) and "Description" in value:
                topics.append({
                    "topic": key.replace("1.", "").replace("2.", "").strip(),
                    "roadmap_level": level_name,
                    "description": value["Description"]
                })
    _topics_df = pd.DataFrame(topics)
    return _topics_df

def load_model_and_assets():
    global _model, _qid_encoder, _MAX_LEN
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if _qid_encoder is None:
        _qid_encoder = joblib.load(os.path.join(EXPORT_DIR, "qid_encoder.pkl"))
    if _MAX_LEN is None:
        with open(os.path.join(EXPORT_DIR, "MAX_LEN.txt")) as f:
            _MAX_LEN = int(f.read().strip())

def map_cefr_and_label(p_correct: float):
    if p_correct >= 0.85:
        return "A1", "Easy"
    elif p_correct >= 0.70:
        return "A2", "Medium"
    elif p_correct >= 0.55:
        return "B1", "Medium"
    elif p_correct >= 0.40:
        return "B2", "Hard"
    elif p_correct >= 0.25:
        return "C1", "Hard"
    else:
        return "C2", "Very Hard"

def predict_mastery(user_results: list[float]) -> float:
    if not user_results:
        return 0.5
    return float(np.mean(user_results))

def get_topic_info(current_topic: str, user_results: list[float]):
    load_model_and_assets()
    topics_df = _load_topics_df()

    p_mastery = predict_mastery(user_results)
    cefr, difficulty_label = map_cefr_and_label(p_mastery)

    topic_row = topics_df[topics_df["topic"] == current_topic]
    if len(topic_row) == 0:
        roadmap_level = "Unknown"
        desc = "Topic not found in roadmap."
    else:
        roadmap_level = topic_row["roadmap_level"].iloc[0]
        desc = topic_row["description"].iloc[0]

    return {
        "topic": current_topic,
        "predicted_mastery": round(p_mastery, 3),
        "model_level": cefr,
        "difficulty_label": difficulty_label,
        "roadmap_level": roadmap_level,
        "description": desc
    }
