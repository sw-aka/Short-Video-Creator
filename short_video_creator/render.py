import os
from pathlib import Path
import subprocess

import imageio_ffmpeg

from .errors import TranscriptionError
from .settings import Settings


def extract_audio(input_path: Path, audio_path: Path) -> None:
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
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        raise TranscriptionError(f"Audio extraction failed: {error}") from error
    if result.returncode != 0:
        raise TranscriptionError(f"Audio extraction failed:\n{result.stderr}")


def build_filter_complex(fps, video_stream_index: int, settings: Settings = Settings()) -> str:
    width, height = settings.full_resolution
    main_height = round(height * (settings.percent_main_clip / 100))
    background_height = height - main_height
    return ";".join(
        [
            f"[0:{video_stream_index}]fps={fps},scale={width}:{main_height}:"
            f"force_original_aspect_ratio=increase,crop={width}:{main_height},setsar=1[main]",
            f"[1:v]fps={fps},crop=trunc(iw*0.9/2)*2:trunc(ih/2)*2,scale={width}:"
            f"{background_height}:force_original_aspect_ratio=increase,crop={width}:"
            f"{background_height},setsar=1[background]",
            "[main][background]vstack=inputs=2[stacked]",
            "[stacked]subtitles=filename=captions.ass:fontsdir=.[output]",
        ]
    )


def render_video(
    input_path: Path,
    background_path: Path,
    background_start: int,
    duration: float,
    filter_script_path: Path,
    output_path: Path,
    codec: str,
    working_directory: Path,
    settings: Settings = Settings(),
) -> subprocess.CompletedProcess:
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
        settings.video_bitrate,
        "-c:a",
        "aac",
    ]
    if codec == "libx264":
        command.extend(["-preset", "veryfast", "-threads", str(settings.num_threads)])
    command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", os.fspath(output_path)])
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(120, duration * 10),
        cwd=os.fspath(working_directory),
    )
