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
from PIL import Image, ImageDraw, ImageFont

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


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.WARNING),
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


class Tools:
    @staticmethod
    def round_down(num: float, decimals: int = 0) -> float:
        """Round down a number to the specified number of decimal places."""
        return math.floor(num * 10 ** decimals) / 10 ** decimals


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
    }


def group_caption_segments(timestamps, clip_duration):
    """Group word timestamps using the original caption timing behavior."""
    if not timestamps:
        return []

    segments = []
    previous_time = 0
    queued_texts = []
    full_start = None

    for pos, timestamp in enumerate(timestamps):
        start, end = timestamp["timestamp"]
        text = timestamp["text"]

        if pos + 1 < len(timestamps):
            next_timestamp_start = timestamps[pos + 1]["timestamp"][0]
            if next_timestamp_start > end:
                end = min(end + 0.5, next_timestamp_start)

        if end - previous_time < 0.3 and pos + 1 < len(timestamps):
            if full_start is None:
                full_start = start
            queued_texts.append(text)
            continue

        queued_texts.append(text)
        text = " ".join(queued_texts)
        queued_texts = []

        if full_start is None:
            full_start = start

        if full_start < clip_duration:
            segments.append((full_start, min(end, clip_duration), text))

        previous_time = end
        full_start = None

    return segments


def create_text_image(text, font_path, font_size, max_width):
    """Render a transparent caption image with the configured font."""
    image = Image.new("RGBA", (max_width, font_size * 10), (0, 0, 0, 0))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)
    _, _, width, height = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((max_width - width) / 2, round(height * 0.2)),
        text,
        font=font,
        fill="white",
        stroke_width=FONT_BORDER_WEIGHT,
        stroke_fill="black",
    )
    return image.crop((0, 0, max_width, round(height * 1.6)))


def create_caption_sprite(caption_segments, sprite_path):
    """Render captions into uniform rows of one transparent sprite sheet."""
    caption_images = [
        create_text_image(
            text,
            Path(FONTS_DIR) / FONT_NAME,
            FONT_SIZE,
            FULL_RESOLUTION[0],
        )
        for _, _, text in caption_segments
    ]
    row_height = max((image.height for image in caption_images), default=1)
    sprite = Image.new(
        "RGBA",
        (FULL_RESOLUTION[0], row_height * max(1, len(caption_images))),
        (0, 0, 0, 0),
    )
    for pos, image in enumerate(caption_images):
        sprite.paste(image, (0, pos * row_height), image)
    sprite.save(sprite_path)
    return row_height


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
        background_path = Path(BACKGROUND_VIDEOS_DIR) / background_name
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
    start_time = Tools.round_down(random.uniform(0, metadata["duration"] - duration))
    return background_path, start_time


def build_filter_complex(fps, video_stream_index, caption_segments, caption_row_height):
    """Build the stacked-video and timed-caption ffmpeg filter graph."""
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
    ]

    current_label = "stacked"
    caption_y = round(FULL_RESOLUTION[1] * (TEXT_POSITION_PERCENT / 100))
    for pos, (start, end, _) in enumerate(caption_segments):
        caption_label = f"caption{pos}"
        output_label = f"captioned{pos}"
        filters.append(
            f"[2:v]crop={FULL_RESOLUTION[0]}:{caption_row_height}:"
            f"0:{pos * caption_row_height}[{caption_label}]"
        )
        filters.append(
            f"[{current_label}][{caption_label}]overlay=x=0:y={caption_y}:"
            f"eof_action=pass:shortest=0:repeatlast=1:"
            f"enable='between(t,{start:.6f},{end:.6f})'[{output_label}]"
        )
        current_label = output_label

    return ";".join(filters), current_label


def render_video(
    input_path,
    background_path,
    background_start,
    duration,
    fps,
    sprite_path,
    filter_script_path,
    output_path,
    codec,
):
    """Render a complete output with one ffmpeg invocation."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-y",
        "-i",
        os.fspath(input_path),
        "-ss",
        str(background_start),
        "-t",
        f"{duration:.6f}",
        "-i",
        os.fspath(background_path),
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        os.fspath(sprite_path),
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
    )


def start_process(file_name):
    """Process a video file by applying transformations and saving the output."""
    logging.info(f"Processing: {file_name}")
    start_time = time.time()
    input_path = Path(INPUT_VIDEOS_DIR) / file_name
    output_path = Path(OUTPUT_VIDEOS_DIR) / file_name
    temporary_output_path = None

    try:
        output_path.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".tmp.mp4",
            dir=output_path.parent,
            delete=False,
        ) as temporary_output:
            temporary_output_path = Path(temporary_output.name)

        main_metadata = probe_video(input_path)
        duration = main_metadata["duration"]
        fps = main_metadata["fps"]
        background_path, background_start = select_background(duration)
        render_duration = float(math.floor(Fraction(str(duration)) * fps) / fps)

        with tempfile.TemporaryDirectory(prefix="svc-") as temporary_directory:
            temporary_path = Path(temporary_directory)
            audio_path = temporary_path / "audio.wav"
            sprite_path = temporary_path / "captions.png"
            filter_script_path = temporary_path / "filters.txt"
            extract_audio(input_path, audio_path)
            timestamps = transcriber.transcribe_words(os.fspath(audio_path))
            caption_segments = group_caption_segments(timestamps, duration)
            caption_row_height = create_caption_sprite(caption_segments, sprite_path)
            filter_complex, video_label = build_filter_complex(
                fps,
                main_metadata["video_stream_index"],
                caption_segments,
                caption_row_height,
            )
            filter_script_path.write_text(
                f"{filter_complex};[{video_label}]null[output]",
                encoding="utf-8",
            )

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
                        fps,
                        sprite_path,
                        filter_script_path,
                        temporary_output_path,
                        codec,
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
    Path(INPUT_VIDEOS_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_VIDEOS_DIR).mkdir(parents=True, exist_ok=True)

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
