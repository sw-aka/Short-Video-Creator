import subprocess
import sys

import imageio_ffmpeg

from .settings import Settings

_VIDEO_CODEC_CACHE: dict[str | None, str] = {}


def select_video_codec(settings: Settings = Settings()) -> str:
    override = settings.video_codec
    if override in _VIDEO_CODEC_CACHE:
        return _VIDEO_CODEC_CACHE[override]
    if override:
        _VIDEO_CODEC_CACHE[override] = override
        return override

    if sys.platform == "darwin":
        candidates = ("h264_videotoolbox",)
    elif sys.platform == "win32":
        candidates = ("h264_nvenc", "h264_qsv", "h264_amf")
    elif sys.platform.startswith("linux"):
        candidates = ("h264_nvenc", "h264_qsv")
    else:
        candidates = ()

    selected = "libx264"
    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg encoder probe failed")
        available = {
            parts[1]
            for line in result.stdout.splitlines()
            if len(parts := line.split()) >= 2
        }
        for candidate in candidates:
            if candidate not in available:
                continue
            test = subprocess.run(
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
                    candidate,
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if test.returncode == 0:
                selected = candidate
                break
    except Exception:
        selected = "libx264"

    _VIDEO_CODEC_CACHE[override] = selected
    return selected


def invalidate_video_codec(codec: str, settings: Settings = Settings()) -> None:
    if codec != "libx264" and _VIDEO_CODEC_CACHE.get(settings.video_codec) == codec:
        _VIDEO_CODEC_CACHE[settings.video_codec] = "libx264"
