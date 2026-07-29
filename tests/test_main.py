import json
import runpy
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import pytest

import config
import main


def word(start, end, text):
    return {"timestamp": (start, end), "text": text}


def successful_run(stdout="", stderr=""):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)


def configure_valid_preflight(monkeypatch, tmp_path):
    fonts = tmp_path / "fonts"
    backgrounds = tmp_path / "backgrounds"
    fonts.mkdir()
    backgrounds.mkdir()
    (fonts / "font.ttf").touch()
    (backgrounds / "background.mp4").touch()
    monkeypatch.setattr(main, "FONTS_DIR", fonts)
    monkeypatch.setattr(main, "FONT_NAME", "font.ttf")
    monkeypatch.setattr(main, "BACKGROUND_VIDEOS_DIR", backgrounds)
    monkeypatch.setattr(main.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: successful_run(" .. subtitles       Render text\n"),
    )


@pytest.fixture(autouse=True)
def reset_video_codec_cache(monkeypatch):
    monkeypatch.setattr(main, "_VIDEO_CODEC_CACHE", None)


def test_group_caption_segments_merges_queued_short_words():
    timestamps = [
        word(0.0, 0.1, "one"),
        word(0.1, 0.2, "two"),
        word(0.2, 0.7, "three"),
    ]

    assert main.group_caption_segments(timestamps, 2.0) == [
        (0.0, 0.7, "one two three")
    ]


def test_group_caption_segments_queues_short_word_until_next_segment():
    timestamps = [word(0.0, 0.2, "short"), word(0.2, 0.6, "enough")]

    assert main.group_caption_segments(timestamps, 2.0) == [
        (0.0, 0.6, "short enough")
    ]


def test_group_caption_segments_clamps_final_word_to_clip_duration():
    assert main.group_caption_segments([word(0.8, 1.4, "last")], 1.0) == [
        (0.8, 1.0, "last")
    ]


def test_group_caption_segments_excludes_words_at_or_after_clip_end():
    timestamps = [
        word(0.0, 0.5, "inside"),
        word(1.0, 1.2, "edge"),
        word(1.1, 1.3, "outside"),
    ]

    assert main.group_caption_segments(timestamps, 1.0) == [
        (0.0, 1.0, "inside")
    ]


def test_group_caption_segments_accepts_empty_input():
    assert main.group_caption_segments([], 1.0) == []


def test_escape_ass_text_sanitizes_control_characters():
    assert main.escape_ass_text("{one}\\two\r\nthree\nfour") == (
        r"\{one\} two\Nthree\Nfour"
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0:00:00.00"),
        (61.239, "0:01:01.24"),
        (3599.996, "1:00:00.00"),
        (-1, "0:00:00.00"),
    ],
)
def test_format_ass_timestamp(seconds, expected):
    assert main.format_ass_timestamp(seconds) == expected


def test_write_ass_captions_writes_complete_ass_file(monkeypatch, tmp_path):
    fake_font = SimpleNamespace(getname=lambda: ("Test Font Family", "Regular"))
    truetype_calls = []

    def fake_truetype(path, size):
        truetype_calls.append((path, size))
        return fake_font

    monkeypatch.setattr(main.ImageFont, "truetype", fake_truetype)
    subtitle_path = tmp_path / "captions.ass"
    font_path = tmp_path / "font.ttf"

    main.write_ass_captions(
        [(0.0, 1.25, "First"), (1.25, 2.0, "Second {line}")],
        subtitle_path,
        font_path,
    )

    contents = subtitle_path.read_text(encoding="utf-8")
    assert truetype_calls == [(font_path, main.FONT_SIZE)]
    assert "[Script Info]" in contents
    assert "[V4+ Styles]" in contents
    assert "[Events]" in contents
    assert f"Style: Default,Test Font Family,{main.FONT_SIZE}," in contents
    assert "Dialogue: 0,0:00:00.00,0:00:01.25,Default,,0,0,0,,First" in contents
    assert (
        r"Dialogue: 0,0:00:01.25,0:00:02.00,Default,,0,0,0,,Second \{line\}"
        in contents
    )


def test_build_filter_complex_connects_labels_and_layout(monkeypatch):
    monkeypatch.setattr(main, "FULL_RESOLUTION", (1080, 1920))
    monkeypatch.setattr(main, "PERCENT_MAIN_CLIP", 40)

    graph = main.build_filter_complex(Fraction(30000, 1001), 2)

    assert "[0:2]fps=30000/1001,scale=1080:768" in graph
    assert "[main]" in graph
    assert "[1:v]fps=30000/1001" in graph
    assert "scale=1080:1152" in graph
    assert "[background]" in graph
    assert "[main][background]vstack=inputs=2[stacked]" in graph
    assert "[stacked]subtitles=filename=captions.ass:fontsdir=.[output]" in graph


def test_probe_video_parses_ffprobe_json(monkeypatch, tmp_path):
    metadata = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001",
                "disposition": {"attached_pic": 0},
            },
            {"index": 1, "codec_type": "audio"},
        ],
        "format": {"duration": "12.345"},
    }
    monkeypatch.setattr(main.shutil, "which", lambda command: "ffprobe")
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: successful_run(json.dumps(metadata)),
    )

    result = main.probe_video(tmp_path / "video.mp4")

    assert result == {
        "duration": 12.345,
        "width": 1920,
        "height": 1080,
        "fps": Fraction(30000, 1001),
        "video_stream_index": 0,
        "has_audio": True,
    }


def test_probe_video_parses_ffmpeg_stderr_fallback(monkeypatch, tmp_path):
    stderr = """
Duration: 00:01:02.50, start: 0.000000, bitrate: 1000 kb/s
Stream #0:0: Video: h264, yuv420p, 1280x720, 29.97 fps, 29.97 tbr
Stream #0:1: Audio: aac, 48000 Hz, stereo
"""
    monkeypatch.setattr(main.shutil, "which", lambda command: None)
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", stderr),
    )

    result = main.probe_video(tmp_path / "video.mp4")

    assert result == {
        "duration": 62.5,
        "width": 1280,
        "height": 720,
        "fps": Fraction("29.97"),
        "video_stream_index": 0,
        "has_audio": True,
    }


def test_select_background_filters_ineligible_videos(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "BACKGROUND_VIDEOS_DIR", tmp_path)
    monkeypatch.setattr(main, "list_video_files", lambda directory: ["short.mp4", "long.mp4"])
    monkeypatch.setattr(
        main,
        "probe_video",
        lambda path: {"duration": 5.0 if path.name == "short.mp4" else 12.0},
    )
    monkeypatch.setattr(main.random, "choice", lambda choices: choices[0])
    monkeypatch.setattr(main.random, "uniform", lambda start, end: end)
    main._BACKGROUND_METADATA_CACHE.clear()

    background, start = main.select_background(10.0)

    assert background == (tmp_path / "long.mp4").resolve()
    assert start == 2


def test_select_background_errors_when_none_are_long_enough(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "BACKGROUND_VIDEOS_DIR", tmp_path)
    monkeypatch.setattr(main, "list_video_files", lambda directory: ["short.mp4"])
    monkeypatch.setattr(main, "probe_video", lambda path: {"duration": 2.0})
    main._BACKGROUND_METADATA_CACHE.clear()

    with pytest.raises(ValueError, match="No background video is at least 3.000 seconds"):
        main.select_background(3.0)


@pytest.mark.parametrize("value", [-1, 0, 100, 101, True])
def test_validate_preflight_rejects_main_clip_percent(monkeypatch, tmp_path, value):
    configure_valid_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "PERCENT_MAIN_CLIP", value)

    with pytest.raises(ValueError, match="PERCENT_MAIN_CLIP"):
        main.validate_preflight()


def test_validate_preflight_rejects_zero_height_section(monkeypatch, tmp_path):
    configure_valid_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "FULL_RESOLUTION", (1080, 2))
    monkeypatch.setattr(main, "PERCENT_MAIN_CLIP", 1)

    with pytest.raises(ValueError, match="heights of at least 2"):
        main.validate_preflight()


def test_validate_preflight_reports_missing_font(monkeypatch, tmp_path):
    configure_valid_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "FONT_NAME", "missing.ttf")

    with pytest.raises(ValueError, match="Font file does not exist"):
        main.validate_preflight()


@pytest.mark.parametrize(
    "resolution",
    [
        "1080x1920",
        (1080,),
        (1080.0, 1920),
        (True, 1920),
        (1080, 1919),
        (0, 1920),
        (-2, 1920),
    ],
)
def test_validate_preflight_rejects_invalid_resolution(monkeypatch, tmp_path, resolution):
    configure_valid_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "FULL_RESOLUTION", resolution)

    with pytest.raises(ValueError, match="two positive even integers"):
        main.validate_preflight()


@pytest.mark.parametrize("value", [-1, 101])
def test_validate_preflight_rejects_text_position_bounds(monkeypatch, tmp_path, value):
    configure_valid_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "TEXT_POSITION_PERCENT", value)

    with pytest.raises(ValueError, match="TEXT_POSITION_PERCENT must be between 0 and 100"):
        main.validate_preflight()


@pytest.mark.parametrize("name", ["MAX_NUMBER_OF_PROCESSES", "NUM_THREADS"])
def test_validate_preflight_rejects_nonpositive_worker_counts(monkeypatch, tmp_path, name):
    configure_valid_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(main, name, 0)

    with pytest.raises(ValueError, match=rf"{name} must be an integer of at least 1"):
        main.validate_preflight()


def test_validate_preflight_reports_missing_background_directory(monkeypatch, tmp_path):
    configure_valid_preflight(monkeypatch, tmp_path)
    missing_directory = tmp_path / "missing-backgrounds"
    monkeypatch.setattr(main, "BACKGROUND_VIDEOS_DIR", missing_directory)

    with pytest.raises(ValueError, match="Background video directory does not exist"):
        main.validate_preflight()


def test_validate_preflight_reports_empty_background_directory(monkeypatch, tmp_path):
    configure_valid_preflight(monkeypatch, tmp_path)
    empty_directory = tmp_path / "empty-backgrounds"
    empty_directory.mkdir()
    monkeypatch.setattr(main, "BACKGROUND_VIDEOS_DIR", empty_directory)

    with pytest.raises(ValueError, match="No video files found"):
        main.validate_preflight()


def test_validate_preflight_reports_ffmpeg_filter_probe_failure(monkeypatch, tmp_path):
    configure_valid_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", "probe failed"),
    )

    with pytest.raises(ValueError, match="Unable to inspect ffmpeg filters: probe failed"):
        main.validate_preflight()


def test_validate_preflight_requires_subtitles_filter(monkeypatch, tmp_path):
    configure_valid_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: successful_run(" V->A volume Adjust volume\n"),
    )

    with pytest.raises(ValueError, match="required subtitles filter"):
        main.validate_preflight()


def test_select_video_codec_returns_configured_override(monkeypatch):
    monkeypatch.setattr(main, "VIDEO_CODEC", "configured_codec")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("configured codec should not invoke ffmpeg")

    monkeypatch.setattr(main.subprocess, "run", unexpected_run)

    assert main.select_video_codec() == "configured_codec"
    assert main.select_video_codec() == "configured_codec"


def test_select_video_codec_caches_successful_hardware_candidate(monkeypatch):
    monkeypatch.setattr(main, "VIDEO_CODEC", None)
    monkeypatch.setattr(main.sys, "platform", "linux")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "-encoders" in command:
            return successful_run(" V..... h264_nvenc NVIDIA encoder\n")
        return successful_run()

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    assert main.select_video_codec() == "h264_nvenc"
    assert main.select_video_codec() == "h264_nvenc"
    assert len(commands) == 2
    assert "color=size=64x64:duration=0.1" in commands[1]


def test_select_video_codec_falls_back_when_test_encodes_fail(monkeypatch):
    monkeypatch.setattr(main, "VIDEO_CODEC", None)
    monkeypatch.setattr(main.sys, "platform", "linux")
    tested_codecs = []

    def fake_run(command, **kwargs):
        if "-encoders" in command:
            return successful_run(
                " V..... h264_nvenc NVIDIA encoder\n V..... h264_qsv Intel encoder\n"
            )
        tested_codecs.append(command[command.index("-c:v") + 1])
        return subprocess.CompletedProcess(command, 1, "", "encode failed")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    assert main.select_video_codec() == "libx264"
    assert tested_codecs == ["h264_nvenc", "h264_qsv"]


def test_invalidate_video_codec_switches_cached_hardware_codec(monkeypatch):
    monkeypatch.setattr(main, "VIDEO_CODEC", None)
    monkeypatch.setattr(main.sys, "platform", "darwin")
    call_count = 0

    def fake_run(command, **kwargs):
        nonlocal call_count
        call_count += 1
        if "-encoders" in command:
            return successful_run(" V..... h264_videotoolbox VideoToolbox encoder\n")
        return successful_run()

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    assert main.select_video_codec() == "h264_videotoolbox"
    main.invalidate_video_codec("h264_videotoolbox")

    assert main.select_video_codec() == "libx264"
    assert call_count == 2


def test_start_process_retries_software_codec_and_invalidates_failure(
    monkeypatch, tmp_path
):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    fonts = tmp_path / "fonts"
    inputs.mkdir()
    outputs.mkdir()
    fonts.mkdir()
    (inputs / "clip.mp4").touch()
    (fonts / "font.ttf").touch()
    monkeypatch.setattr(main, "INPUT_VIDEOS_DIR", inputs)
    monkeypatch.setattr(main, "OUTPUT_VIDEOS_DIR", outputs)
    monkeypatch.setattr(main, "FONTS_DIR", fonts)
    monkeypatch.setattr(main, "FONT_NAME", "font.ttf")
    monkeypatch.setattr(
        main,
        "probe_video",
        lambda path: {
            "duration": 2.0,
            "fps": Fraction(30),
            "video_stream_index": 0,
            "has_audio": True,
        },
    )
    monkeypatch.setattr(main, "select_background", lambda duration: (tmp_path / "bg.mp4", 0))
    monkeypatch.setattr(main, "extract_audio", lambda *args: None)
    monkeypatch.setattr(main.transcriber, "transcribe_words", lambda path: [])
    monkeypatch.setattr(main, "write_ass_captions", lambda *args: None)
    monkeypatch.setattr(main, "build_filter_complex", lambda *args: "filter graph")
    monkeypatch.setattr(main, "select_video_codec", lambda: "h264_test")
    invalidated = []
    monkeypatch.setattr(main, "invalidate_video_codec", invalidated.append)
    codecs = []

    def fake_render(*args):
        codec = args[6]
        codecs.append(codec)
        return subprocess.CompletedProcess([], 1 if codec == "h264_test" else 0, "", "failed")

    monkeypatch.setattr(main, "render_video", fake_render)

    main.start_process("clip.mp4")

    assert codecs == ["h264_test", "libx264"]
    assert invalidated == ["h264_test"]
    assert (outputs / "clip.mp4").is_file()


def test_main_exits_one_when_preflight_fails(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    backgrounds = tmp_path / "backgrounds"
    fonts = tmp_path / "fonts"
    backgrounds.mkdir()
    fonts.mkdir()
    (backgrounds / "background.mp4").touch()
    monkeypatch.setattr(config, "INPUT_VIDEOS_DIR", inputs)
    monkeypatch.setattr(config, "OUTPUT_VIDEOS_DIR", outputs)
    monkeypatch.setattr(config, "BACKGROUND_VIDEOS_DIR", backgrounds)
    monkeypatch.setattr(config, "FONTS_DIR", fonts)
    monkeypatch.setattr(config, "FONT_NAME", "missing.ttf")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: successful_run(" .. subtitles       Render text\n"),
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(main.__file__, run_name="__main__")

    assert exit_info.value.code == 1
