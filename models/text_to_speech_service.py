# models/text_to_speech_service.py
import os
import uuid
from gtts import gTTS

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "tts_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)

def synthesize_tts(text: str, lang: str = "en"):
    """
    Generate mp3 file and return full path.
    """
    tts = gTTS(text=text, lang=lang, slow=False)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(CACHE_DIR, filename)
    tts.save(out_path)
    return out_path
