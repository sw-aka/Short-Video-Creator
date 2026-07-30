from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_name: str = "nemo-parakeet-tdt-0.6b-v3"
    quantization: str = "int8"
    num_threads: int = 12
    font_size: int = 100
    font_border_weight: int = 10
    full_resolution: tuple[int, int] = (1080, 1920)
    percent_main_clip: float = 40
    text_position_percent: float = 30
    video_codec: str | None = None
    video_bitrate: str = "8M"
    font_path: Path | None = None
