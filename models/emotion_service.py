# models/emotion_service.py
import os
import cv2
import numpy as np
from keras.models import model_from_json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_json_path = os.path.join(DATA_DIR, "emotiondetector.json")
_h5_path   = os.path.join(DATA_DIR, "emotiondetector.h5")

_model = None
_face_cascade = None
_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def _load_model():
    global _model, _face_cascade
    if _model is None:
        with open(_json_path, "r") as jf:
            model_json = jf.read()
        _model = model_from_json(model_json)
        _model.load_weights(_h5_path)
    if _face_cascade is None:
        haar = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(haar)

def _prep_face(gray48):
    eq = cv2.equalizeHist(gray48)
    feat = eq.astype("float32") / 255.0
    feat = np.expand_dims(feat, axis=(0, -1))  # (1,48,48,1)
    return feat

def predict_emotion_from_frame(bgr_image: np.ndarray):
    _load_model()
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(80,80))
    if len(faces) == 0:
        faces = _face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60,60))
        if len(faces) == 0:
            return None, None
    (x,y,w,h) = max(faces, key=lambda b: b[2]*b[3])
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48,48), interpolation=cv2.INTER_AREA)
    feats = _prep_face(face)
    pred = _model.predict(feats, verbose=0)
    label = _labels[int(np.argmax(pred))]
    return label, (x,y,w,h)
