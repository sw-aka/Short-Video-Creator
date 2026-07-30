from contextlib import contextmanager
import logging
import os
from pathlib import Path
import wave

import onnx_asr

from .errors import TranscriptionError
from .settings import Settings

logger = logging.getLogger(__name__)
MAX_WORD_DURATION = 0.5
_MODELS: dict[tuple[str, str], object] = {}


@contextmanager
def _suppress_stderr():
    """Suppress C-library stderr output unless debug logging is enabled."""
    if os.environ.get("LOG_LEVEL", "WARNING").upper() == "DEBUG":
        yield
        return

    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def _get_model(settings: Settings = Settings()):
    key = (settings.model_name, settings.quantization)
    if key not in _MODELS:
        logger.info("Loading ASR model: %s (%s)", *key)
        if os.environ.get("LOG_LEVEL", "WARNING").upper() != "DEBUG":
            import onnxruntime

            onnxruntime.set_default_logger_severity(3)
        try:
            with _suppress_stderr():
                _MODELS[key] = onnx_asr.load_model(
                    settings.model_name, quantization=settings.quantization
                ).with_timestamps()
        except Exception as error:
            raise TranscriptionError(f"Unable to load ASR model: {error}") from error
    return _MODELS[key]


def _merge_tokens_into_words(tokens, timestamps, clip_duration=None):
    words = []
    for token, start in zip(tokens, timestamps):
        if token.startswith(" ") or not words:
            words.append({"text": token.strip(), "start": start})
        else:
            words[-1]["text"] += token
    words = [word for word in words if word["text"]]
    results = []
    for pos, word in enumerate(words):
        start = word["start"]
        if pos + 1 < len(words):
            end = min(words[pos + 1]["start"], start + MAX_WORD_DURATION)
        else:
            end = start + MAX_WORD_DURATION
            if clip_duration is not None:
                end = min(end, clip_duration)
        results.append({"timestamp": (start, end), "text": word["text"]})
    return results


def transcribe_words(audio_path: str | Path, settings: Settings = Settings()) -> list[dict]:
    try:
        model = _get_model(settings)
        with _suppress_stderr():
            result = model.recognize(os.fspath(audio_path))
        if not result.text.strip():
            return []
        with wave.open(os.fspath(audio_path), "rb") as audio_file:
            duration = audio_file.getnframes() / audio_file.getframerate()
        return _merge_tokens_into_words(result.tokens, result.timestamps, duration)
    except TranscriptionError:
        raise
    except Exception as error:
        raise TranscriptionError(f"Transcription failed: {error}") from error
