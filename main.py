import concurrent.futures
from fractions import Fraction
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

import imageio_ffmpeg
from PIL import ImageFont

import transcriber
from config import (
    BACKGROUND_VIDEOS_DIR,
    FONT_BORDER_WEIGHT,
    FONTS_DIR,
    FONT_NAME,
    FONT_SIZE,
    FULL_RESOLUTION,
    INPUT_VIDEOS_DIR,
    MAX_NUMBER_OF_PROCESSES,
    OUTPUT_VIDEOS_DIR,
    PERCENT_MAIN_CLIP,
    TEXT_POSITION_PERCENT,
    NUM_THREADS,
    VIDEO_BITRATE,
    VIDEO_CODEC,
)


def configure_logging():
    """Configure logging in the parent or a spawned worker process."""
    logging.basicConfig(
        level=getattr(
            logging,
            os.environ.get("LOG_LEVEL", "INFO").upper(),
            logging.WARNING,
        ),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def list_video_files(directory):
    """Return sorted video file names from a directory."""
    video_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    directory = Path(directory)
    return sorted(
        path.name
        for path in directory.iterdir()
        if not path.name.startswith(".")
        and path.is_file()
        and path.suffix.lower() in video_extensions
    )


def validate_preflight():
    """Raise a clear error for invalid configuration or missing assets."""
    errors = []
    font_path = Path(FONTS_DIR) / FONT_NAME
    if not font_path.is_file():
        errors.append(f"Font file does not exist: {font_path}")

    background_directory = Path(BACKGROUND_VIDEOS_DIR)
    if not background_directory.is_dir():
        errors.append(
            f"Background video directory does not exist: {background_directory}"
        )
    elif not list_video_files(background_directory):
        errors.append(f"No video files found in {background_directory}")

    try:
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        filter_result = subprocess.run(
            [os.fspath(ffmpeg_path), "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if filter_result.returncode != 0:
            details = filter_result.stderr.strip() or f"exit code {filter_result.returncode}"
            errors.append(f"Unable to inspect ffmpeg filters: {details}")
        elif not any(
            len(parts := line.split()) >= 2 and parts[1] == "subtitles"
            for line in filter_result.stdout.splitlines()
        ):
            errors.append("ffmpeg does not provide the required subtitles filter")
    except Exception as error:
        errors.append(f"Unable to inspect ffmpeg filters: {error}")

    resolution_is_valid = not (
        not isinstance(FULL_RESOLUTION, (tuple, list))
        or len(FULL_RESOLUTION) != 2
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            or dimension % 2 != 0
            for dimension in FULL_RESOLUTION
        )
    )
    if not resolution_is_valid:
        errors.append("FULL_RESOLUTION must contain two positive even integers")

    main_percent_is_valid = (
        isinstance(PERCENT_MAIN_CLIP, (int, float))
        and not isinstance(PERCENT_MAIN_CLIP, bool)
        and 0 < PERCENT_MAIN_CLIP < 100
    )
    if not main_percent_is_valid:
        errors.append("PERCENT_MAIN_CLIP must be greater than 0 and less than 100")
    elif resolution_is_valid:
        main_height = round(FULL_RESOLUTION[1] * (PERCENT_MAIN_CLIP / 100))
        background_height = FULL_RESOLUTION[1] - main_height
        if main_height < 2 or background_height < 2:
            errors.append(
                "PERCENT_MAIN_CLIP must produce main and background heights of at least 2"
            )

    if (
        not isinstance(TEXT_POSITION_PERCENT, (int, float))
        or isinstance(TEXT_POSITION_PERCENT, bool)
        or not 0 <= TEXT_POSITION_PERCENT <= 100
    ):
        errors.append("TEXT_POSITION_PERCENT must be between 0 and 100")

    for name, value in (
        ("MAX_NUMBER_OF_PROCESSES", MAX_NUMBER_OF_PROCESSES),
        ("NUM_THREADS", NUM_THREADS),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"{name} must be an integer of at least 1")

    if errors:
        raise ValueError("Preflight validation failed:\n- " + "\n- ".join(errors))


_VIDEO_CODEC_CACHE = None
_BACKGROUND_METADATA_CACHE = {}


def select_video_codec():
    """Return the configured codec or probe ffmpeg once for a platform encoder."""
    global _VIDEO_CODEC_CACHE

    if _VIDEO_CODEC_CACHE is not None:
        return _VIDEO_CODEC_CACHE

    if VIDEO_CODEC:
        _VIDEO_CODEC_CACHE = VIDEO_CODEC
        return _VIDEO_CODEC_CACHE

    if sys.platform == "darwin":
        candidates = ("h264_videotoolbox",)
    elif sys.platform == "win32":
        candidates = ("h264_nvenc", "h264_qsv", "h264_amf")
    elif sys.platform.startswith("linux"):
        candidates = ("h264_nvenc", "h264_qsv")
    else:
        candidates = ()

    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg encoder probe failed")
        available_encoders = {
            parts[1]
            for line in result.stdout.splitlines()
            if len(parts := line.split()) >= 2
        }
        _VIDEO_CODEC_CACHE = "libx264"
        for codec in candidates:
            if codec not in available_encoders:
                continue
            encode_test = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=size=64x64:duration=0.1",
                    "-c:v",
                    codec,
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if encode_test.returncode == 0:
                _VIDEO_CODEC_CACHE = codec
                break
    except Exception:
        _VIDEO_CODEC_CACHE = "libx264"

    return _VIDEO_CODEC_CACHE


def invalidate_video_codec(codec):
    """Stop reusing a hardware codec after a render failure."""
    global _VIDEO_CODEC_CACHE
    if codec != "libx264" and _VIDEO_CODEC_CACHE == codec:
        _VIDEO_CODEC_CACHE = "libx264"


def probe_video(video_path):
    """Return duration, resolution, and frame rate for a video."""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is not None:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                os.fspath(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Unable to probe {video_path}: {result.stderr.strip()}")

        try:
            metadata = json.loads(result.stdout)
            video_stream = next(
                stream
                for stream in metadata.get("streams", [])
                if stream.get("codec_type") == "video"
                and not stream.get("disposition", {}).get("attached_pic", 0)
            )
            fps = Fraction(video_stream["r_frame_rate"])
            duration = float(metadata["format"]["duration"])
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            video_stream_index = int(video_stream["index"])
            has_audio = any(
                stream.get("codec_type") == "audio"
                for stream in metadata.get("streams", [])
            )
            if fps <= 0:
                raise ValueError("frame rate must be positive")
        except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as error:
            raise RuntimeError(f"Unable to probe {video_path}: invalid metadata") from error

        return {
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "video_stream_index": video_stream_index,
            "has_audio": has_audio,
        }

    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", os.fspath(video_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    probe_text = result.stderr

    duration_match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        probe_text,
    )
    video_line = next(
        (
            line
            for line in probe_text.splitlines()
            if "Stream #" in line
            and "Video:" in line
            and "attached pic" not in line.lower()
        ),
        "",
    )
    has_audio = any(
        "Stream #" in line and "Audio:" in line
        for line in probe_text.splitlines()
    )
    stream_index_match = re.search(r"Stream\s+#\d+:(\d+)", video_line)
    resolution_match = re.search(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", video_line)
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps\b", video_line)
    if fps_match is None:
        fps_match = re.search(r"(\d+(?:\.\d+)?)\s+tbr\b", video_line)

    missing_fields = []
    if duration_match is None:
        missing_fields.append("duration")
    if not video_line or stream_index_match is None:
        missing_fields.append("non-attached video stream")
    if resolution_match is None:
        missing_fields.append("resolution")
    if fps_match is None:
        missing_fields.append("fps")
    if missing_fields:
        message = f"Unable to probe {video_path}: missing {', '.join(missing_fields)}"
        logging.error(message)
        raise RuntimeError(message)

    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return {
        "duration": duration,
        "width": int(resolution_match.group(1)),
        "height": int(resolution_match.group(2)),
        "fps": Fraction(fps_match.group(1)),
        "video_stream_index": int(stream_index_match.group(1)),
        "has_audio": has_audio,
    }


def group_caption_segments(timestamps, clip_duration):
    """Group word timestamps using the original caption timing behavior."""
    if not timestamps:
        return []

    segments = []
    previous_time = 0
    queued_texts = []
    queued_end = None
    full_start = None

    for pos, timestamp in enumerate(timestamps):
        start, end = timestamp["timestamp"]
        text = timestamp["text"]

        if start >= clip_duration:
            if queued_texts and full_start is not None:
                segments.append(
                    (
                        full_start,
                        min(queued_end, clip_duration),
                        " ".join(queued_texts),
                    )
                )
            break

        if pos + 1 < len(timestamps):
            next_timestamp_start = timestamps[pos + 1]["timestamp"][0]
            if next_timestamp_start > end:
                end = min(end + 0.5, next_timestamp_start)

        if end - previous_time < 0.3 and pos + 1 < len(timestamps):
            if full_start is None:
                full_start = start
            queued_texts.append(text)
            queued_end = end
            continue

        queued_texts.append(text)
        text = " ".join(queued_texts)
        queued_texts = []
        queued_end = None

        if full_start is None:
            full_start = start

        if full_start < clip_duration:
            segments.append((full_start, min(end, clip_duration), text))

        previous_time = end
        full_start = None

    return segments


def format_ass_timestamp(seconds):
    """Format seconds as an ASS H:MM:SS.cc timestamp."""
    total_centiseconds = max(0, round(seconds * 100))
    total_seconds, centiseconds = divmod(total_centiseconds, 100)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def escape_ass_text(text):
    """Escape text that libass would otherwise interpret as markup."""
    return (
        text.replace("\\", " ")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", r"\N")
        .replace("\r", r"\N")
        .replace("\n", r"\N")
    )


def write_ass_captions(caption_segments, subtitle_path, font_path):
    """Write timed captions using the configured font and placement."""
    font_family = ImageFont.truetype(font_path, FONT_SIZE).getname()[0]
    margin_vertical = round(FULL_RESOLUTION[1] * (TEXT_POSITION_PERCENT / 100))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {FULL_RESOLUTION[0]}",
        f"PlayResY: {FULL_RESOLUTION[1]}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_family},{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,"
        f"&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,"
        f"{FONT_BORDER_WEIGHT},0,8,40,40,{margin_vertical},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    lines.extend(
        "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}".format(
            start=format_ass_timestamp(start),
            end=format_ass_timestamp(end),
            text=escape_ass_text(text),
        )
        for start, end, text in caption_segments
    )
    subtitle_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_audio(input_path, audio_path):
    """Extract the main audio as a 16 kHz mono PCM WAV for ASR."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        os.fspath(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        os.fspath(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr}")


def select_background(duration):
    """Select a random background and a whole-second start offset."""
    eligible_backgrounds = []
    for background_name in list_video_files(BACKGROUND_VIDEOS_DIR):
        background_path = (Path(BACKGROUND_VIDEOS_DIR) / background_name).resolve()
        metadata = _BACKGROUND_METADATA_CACHE.get(background_path)
        if metadata is None:
            metadata = probe_video(background_path)
            _BACKGROUND_METADATA_CACHE[background_path] = metadata
        if metadata["duration"] >= duration:
            eligible_backgrounds.append((background_path, metadata))

    if not eligible_backgrounds:
        raise ValueError(
            f"No background video is at least {duration:.3f} seconds long"
        )

    background_path, metadata = random.choice(eligible_backgrounds)
    start_time = math.floor(random.uniform(0, metadata["duration"] - duration))
    return background_path, start_time


def build_filter_complex(fps, video_stream_index):
    """Build the stacked-video and ASS-caption ffmpeg filter graph."""
    main_height = round(FULL_RESOLUTION[1] * (PERCENT_MAIN_CLIP / 100))
    background_height = FULL_RESOLUTION[1] - main_height
    filters = [
        f"[0:{video_stream_index}]fps={fps},"
        f"scale={FULL_RESOLUTION[0]}:{main_height}:"
        "force_original_aspect_ratio=increase,"
        f"crop={FULL_RESOLUTION[0]}:{main_height},setsar=1[main]",
        "[1:v]"
        "fps={fps},crop=trunc(iw*0.9/2)*2:trunc(ih/2)*2,"
        "scale={width}:{height}:force_original_aspect_ratio=increase,"
        "crop={width}:{height},setsar=1[background]".format(
            fps=fps,
            width=FULL_RESOLUTION[0],
            height=background_height,
        ),
        "[main][background]vstack=inputs=2[stacked]",
        "[stacked]subtitles=filename=captions.ass:fontsdir=.[output]",
    ]
    return ";".join(filters)


def render_video(
    input_path,
    background_path,
    background_start,
    duration,
    filter_script_path,
    output_path,
    codec,
    working_directory,
):
    """Render a complete output with one ffmpeg invocation."""
    command = [
        os.fspath(Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        os.fspath(input_path),
        "-ss",
        str(background_start),
        "-t",
        f"{duration:.6f}",
        "-i",
        os.fspath(background_path),
    ]

    command.extend(
        [
            "-filter_complex_script",
            os.fspath(filter_script_path),
            "-map",
            "[output]",
            "-map",
            "0:a:0",
            "-t",
            f"{duration:.6f}",
            "-c:v",
            codec,
            "-b:v",
            VIDEO_BITRATE,
            "-c:a",
            "aac",
        ]
    )
    if codec == "libx264":
        command.extend(["-preset", "veryfast", "-threads", str(NUM_THREADS)])
    command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", os.fspath(output_path)])

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(120, duration * 10),
        cwd=os.fspath(working_directory),
    )


def start_process(file_name):
    """Process a video file by applying transformations and saving the output."""
    configure_logging()
    logging.info(f"Processing: {file_name}")
    start_time = time.time()
    input_path = (Path(INPUT_VIDEOS_DIR) / file_name).resolve()
    output_path = (Path(OUTPUT_VIDEOS_DIR) / file_name).resolve()
    temporary_output_path = None

    try:
        output_path.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.tmp.",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as temporary_output:
            temporary_output_path = Path(temporary_output.name)

        main_metadata = probe_video(input_path)
        if not main_metadata["has_audio"]:
            raise ValueError(f"{file_name}: no audio stream")
        duration = main_metadata["duration"]
        fps = main_metadata["fps"]
        background_path, background_start = select_background(duration)
        render_duration = float(math.floor(Fraction(str(duration)) * fps) / fps)

        with tempfile.TemporaryDirectory(prefix="svc-") as temporary_directory:
            temporary_path = Path(temporary_directory).resolve()
            audio_path = temporary_path / "audio.wav"
            subtitle_path = temporary_path / "captions.ass"
            filter_script_path = temporary_path / "filters.txt"
            font_path = (Path(FONTS_DIR) / FONT_NAME).resolve()
            shutil.copy2(font_path, temporary_path / f"caption-font{font_path.suffix}")
            extract_audio(input_path, audio_path)
            timestamps = transcriber.transcribe_words(os.fspath(audio_path))
            caption_segments = group_caption_segments(timestamps, duration)
            write_ass_captions(caption_segments, subtitle_path, font_path)
            filter_complex = build_filter_complex(
                fps,
                main_metadata["video_stream_index"],
            )
            filter_script_path.write_text(filter_complex, encoding="utf-8")

            logging.info(f"Saving: {file_name}")
            selected_codec = select_video_codec()
            codecs = [selected_codec]
            if selected_codec != "libx264":
                codecs.append("libx264")

            last_error = None
            for codec in codecs:
                try:
                    result = render_video(
                        input_path,
                        background_path,
                        background_start,
                        render_duration,
                        filter_script_path,
                        temporary_output_path,
                        codec,
                        temporary_path,
                    )
                    if result.returncode == 0:
                        last_error = None
                        break
                    last_error = result.stderr
                except Exception as error:
                    last_error = str(error)
                invalidate_video_codec(codec)
                logging.error(
                    "ERROR Saving: %s. Codec %s failed:\n%s",
                    file_name,
                    codec,
                    last_error,
                )

            if last_error is not None:
                raise RuntimeError(f"Failed to save {file_name}:\n{last_error}")
            temporary_output_path.replace(output_path)
    finally:
        if temporary_output_path is not None:
            try:
                temporary_output_path.unlink(missing_ok=True)
            except OSError as error:
                logging.warning(
                    "Failed to remove temporary output %s: %s",
                    temporary_output_path,
                    error,
                )
        logging.info(f"Runtime: {round(time.time() - start_time, 2)} - {file_name}")


if __name__ == "__main__":
    configure_logging()
    Path(INPUT_VIDEOS_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_VIDEOS_DIR).mkdir(parents=True, exist_ok=True)

    try:
        validate_preflight()
    except ValueError as error:
        logging.error("%s", error)
        sys.exit(1)

    # Only the parent process reads this list, avoiding a queue feeder startup race.
    pending_videos = list_video_files(INPUT_VIDEOS_DIR)
    logging.info("STARTED")

    process_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_NUMBER_OF_PROCESSES,
        mp_context=process_context,
    ) as executor:
        futures = {
            executor.submit(start_process, file_name): file_name
            for file_name in pending_videos
        }
        failed_videos = []
        for future in concurrent.futures.as_completed(futures):
            file_name = futures[future]
            try:
                future.result()
            except Exception:
                failed_videos.append(file_name)
                logging.exception(f"Worker failed: {file_name}")

    if failed_videos:
        logging.error(
            "%d of %d videos failed: %s",
            len(failed_videos),
            len(pending_videos),
            ", ".join(sorted(failed_videos)),
        )
        sys.exit(1)

    logging.info("MAIN PROCESS COMPLETE")
