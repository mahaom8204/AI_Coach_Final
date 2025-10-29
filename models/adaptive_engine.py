# models/adaptive_engine.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.roadmap_loader import flatten_roadmap

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(DATA_DIR, "model")
ROADMAP_PATH = os.path.join(DATA_DIR, "English_Roadmap.json")

_model = None
_qid_encoder = None
_MAX_LEN = None
_topics_df = None

# safe GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

def _load_topics_df():
    global _topics_df
    if _topics_df is not None:
        return _topics_df

    # reuse roadmap_loader.flatten_roadmap() to keep it in sync
    _topics_df = flatten_roadmap()
    return _topics_df

def load_model_and_assets():
    global _model, _qid_encoder, _MAX_LEN
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "dkt_model.h5")
        _model = tf.keras.models.load_model(model_path, compile=False)
    if _qid_encoder is None:
        _qid_encoder = joblib.load(os.path.join(MODEL_DIR, "qid_encoder.pkl"))
    if _MAX_LEN is None:
        with open(os.path.join(MODEL_DIR, "MAX_LEN.txt")) as f:
            _MAX_LEN = int(f.read().strip())

def predict_mastery(user_results):
    """
    user_results is a list of 0/1 correctness from quiz.
    We treat mean correctness as mastery proxy.
    """
    if not user_results:
        return 0.5
    return float(np.mean(user_results))

def map_cefr_and_label(p_correct):
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

def get_topic_info(current_topic: str, user_results: list[int], emotion: str | None):
    """
    Combines mastery + emotion = final difficulty advice.
    """
    load_model_and_assets()
    topics_df = _load_topics_df()

    p_mastery = predict_mastery(user_results)
    cefr, difficulty_label = map_cefr_and_label(p_mastery)

    row = topics_df[topics_df["topic"] == current_topic]
    if len(row) == 0:
        roadmap_level = "Unknown"
        desc = "Topic not found in roadmap."
        examples = []
    else:
        roadmap_level = row["roadmap_level"].iloc[0]
        desc = row["description"].iloc[0]
        examples = row["examples"].iloc[0]

    # emotion-aware adjustment
    if emotion in ["sad", "angry", "disgust", "fear"]:
        tone_msg = "You seem a bit stressed. We'll slow down, simplify explanations, and review basics."
        difficulty_label = "Supportive / Review"
    elif emotion in ["happy", "surprise", "neutral"]:
        tone_msg = "You look engaged. We'll gently increase challenge and push fluency."
        if difficulty_label in ["Easy", "Medium"]:
            difficulty_label = "Stretch Challenge"
    else:
        tone_msg = "Emotion not detected. We'll continue normally."

    return {
        "topic": current_topic,
        "predicted_mastery": round(p_mastery, 3),
        "model_level": cefr,
        "base_difficulty": difficulty_label,
        "roadmap_level": roadmap_level,
        "description": desc,
        "examples": examples,
        "emotion": emotion,
        "coach_message": tone_msg,
    }
