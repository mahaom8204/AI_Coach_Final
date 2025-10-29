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
_labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

def _load_model():
    global _model, _face_cascade
    if _model is None:
        with open(_json_path, "r") as jf:
            model_json = jf.read()
        _model = model_from_json(model_json)
        _model.load_weights(_h5_path)
    if _face_cascade is None:
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(haar_file)

def _extract_features(image_gray48):
    feature = np.array(image_gray48).reshape(1, 48, 48, 1)
    return feature / 255.0

def predict_emotion_from_frame(bgr_image: np.ndarray):
    """
    bgr_image: np.ndarray as decoded by cv2.imdecode
    returns string emotion or None if no face
    """
    _load_model()
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face_img = gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (48,48))
    feats = _extract_features(face_img)
    pred = _model.predict(feats, verbose=0)
    label_idx = int(np.argmax(pred))
    return _labels[label_idx]
