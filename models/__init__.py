# models/__init__.py
"""
Model package for Adaptive English Learning Coach.
Includes modules for:
- adaptive_engine: DKT adaptive learning model
- chatbot_service: Gemini/Gemma conversational tutor
- grammar_checker: fine-tuned T5 grammar correction
- emotion_service: FER2013 emotion detector
- speech_to_text: Whisper transcription
- text_to_speech_service: gTTS text-to-speech
"""

from . import adaptive_engine
from . import chatbot_service
from . import grammar_checker
from . import emotion_service
from . import speech_to_text
from . import text_to_speech_service

__all__ = [
    "adaptive_engine",
    "chatbot_service",
    "grammar_checker",
    "emotion_service",
    "speech_to_text",
    "text_to_speech_service"
]
