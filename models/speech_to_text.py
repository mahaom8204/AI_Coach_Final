import io
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import whisper

_device = "cuda" if torch.cuda.is_available() else "cpu"
_whisper_model = None

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("small", device=_device)

def record_audio(seconds: int = 5, samplerate: int = 16000):
    """
    Record from system default microphone.
    Returns raw WAV bytes.
    """
    st_audio = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    # write to in-memory wav
    buf = io.BytesIO()
    sf.write(buf, st_audio, samplerate, format="WAV")
    return buf.getvalue()

def transcribe_file(file_bytes: bytes, language_code: str = "en"):
    _load_whisper()

    # read from bytes into float32 mono
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    result = _whisper_model.transcribe(
        data,
        language=language_code,
        task="transcribe",
        fp16=False
    )
    return result["text"].strip()
