import os
from gtts import gTTS
import uuid

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tts_cache")
os.makedirs(DATA_DIR, exist_ok=True)

def synthesize_tts(text: str, lang: str = "en"):
    """
    Generate TTS mp3 and return the absolute file path.
    """
    tts = gTTS(text=text, lang=lang, slow=False)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(DATA_DIR, filename)
    tts.save(out_path)
    return out_path
