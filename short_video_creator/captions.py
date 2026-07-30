from pathlib import Path

from PIL import ImageFont

from .settings import Settings


def group_caption_segments(timestamps: list[dict], clip_duration: float) -> list[tuple]:
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
                segments.append((full_start, min(queued_end, clip_duration), " ".join(queued_texts)))
            break
        if pos + 1 < len(timestamps):
            next_start = timestamps[pos + 1]["timestamp"][0]
            if next_start > end:
                end = min(end + 0.5, next_start)
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


def format_ass_timestamp(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * 100))
    total_seconds, centiseconds = divmod(total_centiseconds, 100)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", " ")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", r"\N")
        .replace("\r", r"\N")
        .replace("\n", r"\N")
    )


def write_ass_captions(
    caption_segments: list[tuple],
    subtitle_path: Path,
    font_path: Path,
    settings: Settings = Settings(),
) -> None:
    font_family = ImageFont.truetype(font_path, settings.font_size).getname()[0]
    width, height = settings.full_resolution
    margin_vertical = round(height * (settings.text_position_percent / 100))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_family},{settings.font_size},&H00FFFFFF,&H00FFFFFF,"
        f"&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{settings.font_border_weight},"
        f"0,8,40,40,{margin_vertical},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines.extend(
        f"Dialogue: 0,{format_ass_timestamp(start)},{format_ass_timestamp(end)},"
        f"Default,,0,0,0,,{escape_ass_text(text)}"
        for start, end, text in caption_segments
    )
    subtitle_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
