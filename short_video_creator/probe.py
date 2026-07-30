from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import imageio_ffmpeg

from .errors import ProbeError

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm"})


def list_video_files(directory: str | Path) -> list[str]:
    directory = Path(directory)
    return sorted(
        path.name
        for path in directory.iterdir()
        if not path.name.startswith(".")
        and path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def probe_video(video_path: str | Path) -> dict:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise ProbeError(f"Video file does not exist: {video_path}")

    try:
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
                raise ProbeError(f"Unable to probe {video_path}: {result.stderr.strip()}")
            return _parse_ffprobe(video_path, result.stdout)

        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", os.fspath(video_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return _parse_ffmpeg(video_path, result.stderr)
    except ProbeError:
        raise
    except (OSError, subprocess.SubprocessError) as error:
        raise ProbeError(f"Unable to probe {video_path}: {error}") from error


def _parse_ffprobe(video_path: Path, output: str) -> dict:
    try:
        metadata = json.loads(output)
        video_stream = next(
            stream
            for stream in metadata.get("streams", [])
            if stream.get("codec_type") == "video"
            and not stream.get("disposition", {}).get("attached_pic", 0)
        )
        fps = Fraction(video_stream["r_frame_rate"])
        if fps <= 0:
            raise ValueError("frame rate must be positive")
        return {
            "duration": float(metadata["format"]["duration"]),
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
            "fps": fps,
            "video_stream_index": int(video_stream["index"]),
            "has_audio": any(
                stream.get("codec_type") == "audio" for stream in metadata.get("streams", [])
            ),
        }
    except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as error:
        raise ProbeError(f"Unable to probe {video_path}: invalid metadata") from error


def _parse_ffmpeg(video_path: Path, output: str) -> dict:
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    video_line = next(
        (
            line
            for line in output.splitlines()
            if "Stream #" in line and "Video:" in line and "attached pic" not in line.lower()
        ),
        "",
    )
    stream_match = re.search(r"Stream\s+#\d+:(\d+)", video_line)
    resolution_match = re.search(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", video_line)
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps\b", video_line) or re.search(
        r"(\d+(?:\.\d+)?)\s+tbr\b", video_line
    )
    fields = {
        "duration": duration_match,
        "non-attached video stream": stream_match,
        "resolution": resolution_match,
        "fps": fps_match,
    }
    missing = [name for name, match in fields.items() if match is None]
    if missing:
        raise ProbeError(f"Unable to probe {video_path}: missing {', '.join(missing)}")

    hours, minutes, seconds = duration_match.groups()
    return {
        "duration": int(hours) * 3600 + int(minutes) * 60 + float(seconds),
        "width": int(resolution_match.group(1)),
        "height": int(resolution_match.group(2)),
        "fps": Fraction(fps_match.group(1)),
        "video_stream_index": int(stream_match.group(1)),
        "has_audio": any("Stream #" in line and "Audio:" in line for line in output.splitlines()),
    }
