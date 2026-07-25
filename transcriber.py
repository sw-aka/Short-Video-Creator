from contextlib import contextmanager
import logging
import os
import wave

import onnx_asr

from config import MODEL_NAME, QUANTIZATION

# Maximum duration (in seconds) a single word is assumed to last when the
# model only provides token start times.
MAX_WORD_DURATION = 0.5

# Module-level cache so each worker process loads the model once.
_model = None


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


def _get_model():
    """Load the ASR model, caching it for reuse within the process.

    The model is downloaded from Hugging Face automatically on first use
    and cached locally by huggingface-hub.

    :return: An onnx-asr recognizer that returns timestamped results.
    """
    global _model
    if _model is None:
        logging.info(f"Loading ASR model: {MODEL_NAME} ({QUANTIZATION})")
        if os.environ.get("LOG_LEVEL", "WARNING").upper() != "DEBUG":
            import onnxruntime
            onnxruntime.set_default_logger_severity(3)
        _model = onnx_asr.load_model(MODEL_NAME, quantization=QUANTIZATION).with_timestamps()
        logging.info(f"Loaded ASR model: {MODEL_NAME}")
    return _model


def _merge_tokens_into_words(tokens, timestamps, clip_duration=None):
    """Merge BPE tokens into words with start/end times.

    Tokens beginning with a space start a new word; tokens without a
    leading space (including punctuation such as '.') are appended to the
    current word. The model only provides start times per token, so each
    word's end time is the start of the next word, clamped to at most
    MAX_WORD_DURATION after the word's own start.

    :param tokens: List of BPE token strings (e.g. [' The', ' qu', 'ick']).
    :param timestamps: List of start times in seconds, one per token.
    :param clip_duration: Optional clip end used to clamp the final word.
    :return: A list of dicts: {'timestamp': (start, end), 'text': word}.
    """
    words = []

    for token, start in zip(tokens, timestamps):
        if token.startswith(" ") or len(words) == 0:
            words.append({"text": token.strip(), "start": start})
        else:
            words[-1]["text"] += token

    # Drop words that are empty after stripping (e.g. stray whitespace tokens)
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
        results.append({
            "timestamp": (start, end),
            "text": word["text"],
        })

    return results


def transcribe_words(audio_path):
    """Transcribe an audio file into words with start/end timestamps.

    :param audio_path: Path to a 16 kHz mono WAV file.
    :return: A list of dicts: {'timestamp': (start, end), 'text': word}.
             Returns an empty list if no speech is detected.
    """
    model = _get_model()
    with _suppress_stderr():
        result = model.recognize(audio_path)

    if not result.text.strip():
        return []

    with wave.open(audio_path, "rb") as audio_file:
        clip_duration = audio_file.getnframes() / audio_file.getframerate()

    return _merge_tokens_into_words(result.tokens, result.timestamps, clip_duration)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if len(sys.argv) != 2:
        print("Usage: python transcriber.py <audio file>")
        sys.exit(1)

    for word in transcribe_words(sys.argv[1]):
        start, end = word["timestamp"]
        print(f"{start:7.2f} - {end:7.2f}  {word['text']}")
