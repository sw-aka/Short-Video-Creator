import math
from pathlib import Path
import random

from .errors import PreflightError
from .probe import list_video_files, probe_video

_BACKGROUND_METADATA_CACHE: dict[Path, dict] = {}


def select_background(duration: float, backgrounds_dir: str | Path) -> tuple[Path, int]:
    directory = Path(backgrounds_dir)
    eligible = []
    for background_name in list_video_files(directory):
        background_path = (directory / background_name).resolve()
        metadata = _BACKGROUND_METADATA_CACHE.get(background_path)
        if metadata is None:
            metadata = probe_video(background_path)
            _BACKGROUND_METADATA_CACHE[background_path] = metadata
        if metadata["duration"] >= duration:
            eligible.append((background_path, metadata))
    if not eligible:
        raise PreflightError(f"No background video is at least {duration:.3f} seconds long")
    background_path, metadata = random.choice(eligible)
    start_time = math.floor(random.uniform(0, metadata["duration"] - duration))
    return background_path, start_time
