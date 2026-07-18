"""
Module 1 - Text and Speech
---------------------------
Speech-to-Text via OpenAI Whisper (local, open-source model).
Text-to-Speech via gTTS (Google Text-to-Speech).
"""

import os
import tempfile
import whisper
from gtts import gTTS

_whisper_model = None


def _get_whisper_model(size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(size)
    return _whisper_model


def speech_to_text(audio_filepath: str) -> str:
    """Transcribes a speech audio file (wav/mp3) into text using Whisper."""
    if not audio_filepath:
        return ""
    model = _get_whisper_model()
    result = model.transcribe(audio_filepath)
    return result.get("text", "").strip()


def text_to_speech(text: str, lang: str = "en") -> str:
    """Converts text into a speech audio file and returns the file path."""
    if not text:
        return None
    tts = gTTS(text=text, lang=lang)
    out_path = os.path.join(tempfile.gettempdir(), "response_audio.mp3")
    tts.save(out_path)
    return out_path
