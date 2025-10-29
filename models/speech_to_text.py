import torch
import whisper
import tempfile
import numpy as np
import soundfile as sf  # pip install soundfile

_device = "cuda" if torch.cuda.is_available() else "cpu"
_whisper_model = None

def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        # you used "small" model for multilingual :contentReference[oaicite:18]{index=18}
        _whisper_model = whisper.load_model("small", device=_device)

def transcribe_file(file_bytes: bytes, language_code: str = "en"):
    """
    language_code: "en", "hi", "mr", ...
    """
    load_whisper()

    # Save temp
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Load audio into float32 [-1,1]
    audio, sr = sf.read(tmp_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    result = _whisper_model.transcribe(
        audio,
        language=language_code,
        task="transcribe",
        fp16=False
    )

    return result["text"].strip()
