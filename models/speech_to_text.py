# models/speech_to_text.py
import torch
import whisper
import tempfile
import numpy as np
import soundfile as sf

_device = "cuda" if torch.cuda.is_available() else "cpu"
_whisper_model = None

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        # "small" for multilingual accuracy
        _whisper_model = whisper.load_model("small", device=_device)

def transcribe_file(file_bytes: bytes, language_code: str = "en"):
    """
    file_bytes: audio bytes in WAV format (mono float32 or int16 is fine)
    We'll write to temp, load with soundfile, run inference.
    """
    _load_whisper()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    audio, sr = sf.read(tmp_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    result = _whisper_model.transcribe(
        audio,
        language=language_code,
        task="transcribe",
        fp16=False
    )
    return result["text"].strip()
