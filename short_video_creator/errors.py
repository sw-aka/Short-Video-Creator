class ShortVideoError(Exception):
    """Base class for errors raised by the public library API."""


class PreflightError(ShortVideoError):
    """Required settings, tools, or assets are invalid."""


class ProbeError(ShortVideoError):
    """Video metadata could not be read."""


class NoAudioError(ShortVideoError):
    """The input video has no audio stream."""


class TranscriptionError(ShortVideoError):
    """Audio extraction or speech transcription failed."""


class RenderError(ShortVideoError):
    """The output video could not be rendered."""
