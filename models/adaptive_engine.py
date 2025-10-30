# models/adaptive_engine.py
import os
import joblib
import numpy as np
import tensorflow as tf
from utils.roadmap_loader import flatten_roadmap

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(DATA_DIR, "model")

_model = None
_qid_encoder = None
_MAX_LEN = None
_topics_df = None

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass

def _topics():
    global _topics_df
    if _topics_df is None:
        _topics_df = flatten_roadmap()
    return _topics_df

def load_model_and_assets():
    global _model, _qid_encoder, _MAX_LEN
    if _model is None:
        p = os.path.join(MODEL_DIR, "dkt_model.h5")
        if os.path.exists(p):
            _model = tf.keras.models.load_model(p, compile=False)
    if _qid_encoder is None:
        p = os.path.join(MODEL_DIR, "qid_encoder.pkl")
        if os.path.exists(p):
            _qid_encoder = joblib.load(p)
    if _MAX_LEN is None:
        p = os.path.join(MODEL_DIR, "MAX_LEN.txt")
        if os.path.exists(p):
            with open(p) as f: _MAX_LEN = int(f.read().strip())

def predict_mastery(user_results):
    if not user_results:
        return 0.10  # New user: start basic (A1/Easy)
    return float(np.mean(user_results))

def map_cefr_and_label(p_correct):
    if p_correct < 0.20: return "A1", "Easy"
    if p_correct < 0.40: return "A2", "Easy"
    if p_correct < 0.55: return "B1", "Medium"
    if p_correct < 0.70: return "B2", "Medium"
    if p_correct < 0.85: return "C1", "Hard"
    return "C2", "Very Hard"

def get_topic_info(current_topic: str, user_results: list[int], emotion: str | None):
    load_model_and_assets()
    df = _topics()
    row = df[df["topic"] == current_topic]
    if row.empty:
        roadmap_level, desc, examples = "A1", "Basics kickoff", []
    else:
        roadmap_level = row["roadmap_level"].iloc[0]
        desc         = row["description"].iloc[0]
        examples     = row["examples"].iloc[0]

    p_mastery = predict_mastery(user_results)
    cefr, label = map_cefr_and_label(p_mastery)

    if emotion in ["sad","angry","disgust","fear"]:
        tone = "Supportive pace. Shorter steps, simpler questions."
        label = "Supportive / Review"
    elif emotion in ["happy","surprise","neutral"]:
        tone = "Engaged — gently increasing challenge."
        if label in ["Easy","Medium"]:
            label = "Stretch Challenge"
    else:
        tone = "Emotion not detected. Proceeding at normal pace."

    return {
        "topic": current_topic,
        "predicted_mastery": round(p_mastery, 3),
        "model_level": cefr,
        "base_difficulty": label,
        "roadmap_level": roadmap_level,
        "description": desc,
        "examples": examples,
        "emotion": emotion,
        "coach_message": tone,
    }
