from contextlib import contextmanager, ExitStack
from fractions import Fraction
from importlib.resources import as_file, files
import logging
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import imageio_ffmpeg

from .background import select_background
from .captions import group_caption_segments, write_ass_captions
from .codec import invalidate_video_codec, select_video_codec
from .errors import (
    NoAudioError,
    PreflightError,
    ProbeError,
    RenderError,
    ShortVideoError,
    TranscriptionError,
)
from .probe import list_video_files, probe_video
from .render import build_filter_complex, extract_audio, render_video
from .settings import Settings
from .transcriber import transcribe_words

logger = logging.getLogger(__name__)
_PREFLIGHT_CACHE: set[tuple[Path, Settings, Path]] = set()


@contextmanager
def resolved_font(settings: Settings):
    if settings.font_path is not None:
        try:
            font_path = Path(settings.font_path).expanduser().resolve()
        except (OSError, TypeError, ValueError) as error:
            raise PreflightError(f"Invalid font path: {error}") from error
        yield font_path
        return
    try:
        resource = files("short_video_creator").joinpath(
            "assets", "fonts", "Super Carnival.ttf"
        )
        stack = ExitStack()
        font_path = stack.enter_context(as_file(resource))
    except (OSError, TypeError, ValueError) as error:
        raise PreflightError(f"Unable to resolve bundled font: {error}") from error
    with stack:
        yield font_path


def _validate_settings(settings: Settings) -> list[str]:
    if not isinstance(settings, Settings):
        return ["settings must be a Settings instance"]
    errors = []
    resolution = settings.full_resolution
    resolution_valid = not (
        not isinstance(resolution, tuple)
        or len(resolution) != 2
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            or dimension % 2 != 0
            for dimension in resolution
        )
    )
    if not resolution_valid:
        errors.append("full_resolution must contain two positive even integers")
    percent = settings.percent_main_clip
    if (
        not isinstance(percent, (int, float))
        or isinstance(percent, bool)
        or not 0 < percent < 100
    ):
        errors.append("percent_main_clip must be greater than 0 and less than 100")
    elif resolution_valid:
        main_height = round(resolution[1] * (percent / 100))
        if main_height < 2 or resolution[1] - main_height < 2:
            errors.append("percent_main_clip must produce section heights of at least 2")
    text_position = settings.text_position_percent
    if (
        not isinstance(text_position, (int, float))
        or isinstance(text_position, bool)
        or not 0 <= text_position <= 100
    ):
        errors.append("text_position_percent must be between 0 and 100")
    if (
        not isinstance(settings.num_threads, int)
        or isinstance(settings.num_threads, bool)
        or settings.num_threads < 1
    ):
        errors.append("num_threads must be an integer of at least 1")
    return errors


def preflight(
    backgrounds_dir: str | Path,
    settings: Settings = Settings(),
    *,
    font_path: Path | None = None,
) -> None:
    errors = _validate_settings(settings)
    try:
        hash(settings)
    except TypeError:
        errors.append("settings values must be hashable")
    if errors:
        raise PreflightError("Preflight validation failed:\n- " + "\n- ".join(errors))

    try:
        backgrounds = Path(backgrounds_dir).expanduser().resolve()
    except (OSError, TypeError, ValueError) as error:
        raise PreflightError(f"Invalid background video directory: {error}") from error
    if font_path is None:
        with resolved_font(settings) as resolved:
            return preflight(backgrounds, settings, font_path=resolved)
    try:
        font_path = Path(font_path).expanduser().resolve()
    except (OSError, TypeError, ValueError) as error:
        raise PreflightError(f"Invalid font path: {error}") from error

    key = (backgrounds, settings, font_path)
    if key in _PREFLIGHT_CACHE:
        return
    if not backgrounds.is_dir():
        errors.append(f"Background video directory does not exist: {backgrounds}")
    else:
        try:
            if not list_video_files(backgrounds):
                errors.append(f"No video files found in {backgrounds}")
        except OSError as error:
            errors.append(f"Unable to inspect background directory {backgrounds}: {error}")
    errors.extend(_validate_font_and_ffmpeg(font_path))
    if errors:
        raise PreflightError("Preflight validation failed:\n- " + "\n- ".join(errors))
    select_video_codec(settings)
    _PREFLIGHT_CACHE.add(key)


def _validate_font_and_ffmpeg(font_path: Path) -> list[str]:
    errors = []
    if not font_path.is_file():
        errors.append(f"Font file does not exist: {font_path}")
    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or f"exit code {result.returncode}"
            errors.append(f"Unable to inspect ffmpeg filters: {details}")
        elif not any(
            len(parts := line.split()) >= 2 and parts[1] == "subtitles"
            for line in result.stdout.splitlines()
        ):
            errors.append("ffmpeg does not provide the required subtitles filter")
    except Exception as error:
        errors.append(f"Unable to inspect ffmpeg filters: {error}")
    return errors


def create_short(
    input_video: str | Path,
    output_path: str | Path,
    backgrounds_dir: str | Path,
    settings: Settings = Settings(),
) -> Path:
    if not isinstance(settings, Settings):
        raise PreflightError("settings must be a Settings instance")

    temporary_output_path = None

    try:
        try:
            input_path = Path(input_video).expanduser().resolve()
        except (OSError, TypeError, ValueError) as error:
            raise ProbeError(f"Invalid input video path: {error}") from error
        try:
            backgrounds = Path(backgrounds_dir).expanduser().resolve()
        except (OSError, TypeError, ValueError) as error:
            raise PreflightError(f"Invalid background video directory: {error}") from error
        try:
            destination = Path(output_path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, TypeError, ValueError) as error:
            raise RenderError(f"Unable to prepare output path {output_path!r}: {error}") from error

        with resolved_font(settings) as font_path:
            preflight(backgrounds, settings, font_path=font_path)
            metadata = probe_video(input_path)
            if not metadata["has_audio"]:
                raise NoAudioError(f"{input_path.name}: no audio stream")
            duration = metadata["duration"]
            fps = metadata["fps"]
            background_path, background_start = select_background(duration, backgrounds)
            render_duration = float(math.floor(Fraction(str(duration)) * fps) / fps)
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.stem}.tmp.",
                suffix=destination.suffix,
                dir=destination.parent,
                delete=False,
            ) as temporary_output:
                temporary_output_path = Path(temporary_output.name)

            with tempfile.TemporaryDirectory(prefix="svc-") as temporary_directory:
                working_directory = Path(temporary_directory).resolve()
                audio_path = working_directory / "audio.wav"
                subtitle_path = working_directory / "captions.ass"
                filter_script_path = working_directory / "filters.txt"
                shutil.copy2(font_path, working_directory / f"caption-font{font_path.suffix}")
                extract_audio(input_path, audio_path)
                try:
                    timestamps = transcribe_words(audio_path, settings)
                except TranscriptionError:
                    raise
                except Exception as error:
                    raise TranscriptionError(f"Transcription failed: {error}") from error
                segments = group_caption_segments(timestamps, duration)
                write_ass_captions(segments, subtitle_path, font_path, settings)
                filter_script_path.write_text(
                    build_filter_complex(fps, metadata["video_stream_index"], settings),
                    encoding="utf-8",
                )
                logger.info("Saving: %s", destination.name)
                _render_with_fallback(
                    input_path,
                    background_path,
                    background_start,
                    render_duration,
                    filter_script_path,
                    temporary_output_path,
                    working_directory,
                    settings,
                )
            temporary_output_path.replace(destination)
            temporary_output_path = None
            return destination
    except ShortVideoError:
        raise
    except Exception as error:
        raise RenderError(f"Unable to create short video: {error}") from error
    finally:
        if temporary_output_path is not None:
            try:
                temporary_output_path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("Failed to remove temporary output %s: %s", temporary_output_path, error)


def _render_with_fallback(
    input_path,
    background_path,
    background_start,
    duration,
    filter_script_path,
    output_path,
    working_directory,
    settings,
) -> None:
    selected = select_video_codec(settings)
    codecs = [selected] if selected == "libx264" else [selected, "libx264"]
    last_error = "unknown ffmpeg error"
    for codec in codecs:
        try:
            result = render_video(
                input_path,
                background_path,
                background_start,
                duration,
                filter_script_path,
                output_path,
                codec,
                working_directory,
                settings,
            )
            if result.returncode == 0:
                return
            last_error = result.stderr
        except Exception as error:
            last_error = str(error)
        invalidate_video_codec(codec, settings)
        logger.error("Codec %s failed:\n%s", codec, last_error)
    raise RenderError(f"Failed to render {output_path.name}:\n{last_error}")
