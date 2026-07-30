from contextlib import contextmanager
from dataclasses import FrozenInstanceError, fields
from fractions import Fraction
from importlib.resources import as_file, files
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import pytest

from short_video_creator import (
    NoAudioError,
    PreflightError,
    ProbeError,
    RenderError,
    Settings,
    create_short,
)
from short_video_creator import background, captions, cli, codec, pipeline, probe, render, transcriber


def word(start, end, text):
    return {"timestamp": (start, end), "text": text}


def successful_run(stdout="", stderr=""):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def clear_caches():
    codec._VIDEO_CODEC_CACHE.clear()
    background._BACKGROUND_METADATA_CACHE.clear()
    pipeline._PREFLIGHT_CACHE.clear()


def test_settings_fields_defaults_and_frozen():
    settings = Settings()
    assert [field.name for field in fields(Settings)] == [
        "model_name",
        "quantization",
        "num_threads",
        "font_size",
        "font_border_weight",
        "full_resolution",
        "percent_main_clip",
        "text_position_percent",
        "video_codec",
        "video_bitrate",
        "font_path",
    ]
    assert settings == Settings(
        model_name="nemo-parakeet-tdt-0.6b-v3",
        quantization="int8",
        num_threads=12,
        font_size=100,
        font_border_weight=10,
        full_resolution=(1080, 1920),
        percent_main_clip=40,
        text_position_percent=30,
        video_codec=None,
        video_bitrate="8M",
        font_path=None,
    )
    with pytest.raises(FrozenInstanceError):
        settings.font_size = 1


def test_bundled_font_exists():
    resource = files("short_video_creator").joinpath("assets", "fonts", "Super Carnival.ttf")
    with as_file(resource) as path:
        assert path.is_file()
        assert path.stat().st_size > 0


@pytest.mark.parametrize(
    ("timestamps", "duration", "expected"),
    [
        ([word(0.0, 0.1, "one"), word(0.1, 0.2, "two"), word(0.2, 0.7, "three")], 2, [(0, 0.7, "one two three")]),
        ([word(0, 0.2, "short"), word(0.2, 0.6, "enough")], 2, [(0, 0.6, "short enough")]),
        ([word(0.8, 1.4, "last")], 1, [(0.8, 1, "last")]),
        (
            [
                word(0, 0.5, "inside"),
                word(1, 1.2, "edge"),
                word(1.1, 1.3, "outside"),
            ],
            1,
            [(0, 1, "inside")],
        ),
        ([], 1, []),
    ],
)
def test_group_caption_segments(timestamps, duration, expected):
    assert captions.group_caption_segments(timestamps, duration) == expected


def test_escape_ass_text_sanitizes_control_characters():
    assert captions.escape_ass_text("{one}\\two\r\nthree\nfour") == r"\{one\} two\Nthree\Nfour"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00:00.00"), (61.239, "0:01:01.24"), (3599.996, "1:00:00.00"), (-1, "0:00:00.00")],
)
def test_format_ass_timestamp(seconds, expected):
    assert captions.format_ass_timestamp(seconds) == expected


def test_write_ass_captions_writes_complete_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        captions.ImageFont,
        "truetype",
        lambda path, size: calls.append((path, size)) or SimpleNamespace(getname=lambda: ("Test Font", "Regular")),
    )
    subtitle = tmp_path / "captions.ass"
    font = tmp_path / "font.ttf"
    captions.write_ass_captions([(0, 1.25, "First {line}")], subtitle, font)
    contents = subtitle.read_text(encoding="utf-8")
    assert calls == [(font, 100)]
    assert "[Script Info]" in contents
    assert "[V4+ Styles]" in contents
    assert "[Events]" in contents
    assert "Style: Default,Test Font,100," in contents
    assert r"Dialogue: 0,0:00:00.00,0:00:01.25,Default,,0,0,0,,First \{line\}" in contents


def test_build_filter_complex_connects_layout():
    graph = render.build_filter_complex(Fraction(30000, 1001), 2)
    assert "[0:2]fps=30000/1001,scale=1080:768" in graph
    assert "scale=1080:1152" in graph
    assert "[main][background]vstack=inputs=2[stacked]" in graph
    assert "[stacked]subtitles=filename=captions.ass:fontsdir=.[output]" in graph


def test_probe_video_parses_ffprobe_json(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    metadata = {
        "streams": [
            {"index": 0, "codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30000/1001", "disposition": {"attached_pic": 0}},
            {"index": 1, "codec_type": "audio"},
        ],
        "format": {"duration": "12.345"},
    }
    monkeypatch.setattr(probe.shutil, "which", lambda command: "ffprobe")
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: successful_run(json.dumps(metadata)))
    assert probe.probe_video(video) == {
        "duration": 12.345,
        "width": 1920,
        "height": 1080,
        "fps": Fraction(30000, 1001),
        "video_stream_index": 0,
        "has_audio": True,
    }


def test_probe_video_parses_ffmpeg_fallback(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    stderr = """Duration: 00:01:02.50, start: 0.000000
Stream #0:0: Video: h264, yuv420p, 1280x720, 29.97 fps
Stream #0:1: Audio: aac, 48000 Hz, stereo
"""
    monkeypatch.setattr(probe.shutil, "which", lambda command: None)
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", stderr))
    assert probe.probe_video(video) == {
        "duration": 62.5,
        "width": 1280,
        "height": 720,
        "fps": Fraction("29.97"),
        "video_stream_index": 0,
        "has_audio": True,
    }


def test_select_background_filters_short_videos(monkeypatch, tmp_path):
    short = tmp_path / "short.mp4"
    long = tmp_path / "long.mp4"
    monkeypatch.setattr(background, "list_video_files", lambda directory: [short.name, long.name])
    monkeypatch.setattr(background, "probe_video", lambda path: {"duration": 5 if path.name == "short.mp4" else 12})
    monkeypatch.setattr(background.random, "choice", lambda choices: choices[0])
    monkeypatch.setattr(background.random, "uniform", lambda start, end: end)
    assert background.select_background(10, tmp_path) == (long.resolve(), 2)


def test_select_background_errors_when_none_long_enough(monkeypatch, tmp_path):
    monkeypatch.setattr(background, "list_video_files", lambda directory: ["short.mp4"])
    monkeypatch.setattr(background, "probe_video", lambda path: {"duration": 2})
    with pytest.raises(PreflightError, match="at least 3.000 seconds"):
        background.select_background(3, tmp_path)


def configure_preflight(monkeypatch, tmp_path):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    (backgrounds / "background.mp4").touch()
    font = tmp_path / "font.ttf"
    font.touch()
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: successful_run(" .. subtitles Render text\n"))
    monkeypatch.setattr(pipeline, "select_video_codec", lambda settings: "libx264")
    return backgrounds, font


@pytest.mark.parametrize(
    "settings,match",
    [
        (Settings(percent_main_clip=-1), "percent_main_clip"),
        (Settings(percent_main_clip=0), "percent_main_clip"),
        (Settings(percent_main_clip=100), "percent_main_clip"),
        (Settings(percent_main_clip=101), "percent_main_clip"),
        (Settings(percent_main_clip=True), "percent_main_clip"),
        (Settings(full_resolution=(1080, 2), percent_main_clip=1), "heights of at least 2"),
        (Settings(full_resolution="1080x1920"), "positive even integers"),
        (Settings(full_resolution=(1080,)), "positive even integers"),
        (Settings(full_resolution=(1080.0, 1920)), "positive even integers"),
        (Settings(full_resolution=(True, 1920)), "positive even integers"),
        (Settings(full_resolution=(1080, 1919)), "positive even integers"),
        (Settings(full_resolution=(0, 1920)), "positive even integers"),
        (Settings(full_resolution=(-2, 1920)), "positive even integers"),
        (Settings(text_position_percent=-1), "between 0 and 100"),
        (Settings(text_position_percent=101), "between 0 and 100"),
        (Settings(num_threads=0), "integer of at least 1"),
    ],
)
def test_preflight_rejects_invalid_settings(monkeypatch, tmp_path, settings, match):
    backgrounds, font = configure_preflight(monkeypatch, tmp_path)
    with pytest.raises(PreflightError, match=match):
        pipeline.preflight(backgrounds, settings, font_path=font)


def test_preflight_rejects_list_resolution_before_cache_lookup(monkeypatch, tmp_path):
    backgrounds, font = configure_preflight(monkeypatch, tmp_path)
    settings = Settings(full_resolution=[1080, 1920])
    with pytest.raises(PreflightError, match="positive even integers"):
        pipeline.preflight(backgrounds, settings, font_path=font)


def test_preflight_cache_includes_font_identity(monkeypatch, tmp_path):
    backgrounds, first_font = configure_preflight(monkeypatch, tmp_path)
    second_font = tmp_path / "second.ttf"
    second_font.touch()
    calls = []
    monkeypatch.setattr(
        pipeline,
        "_validate_font_and_ffmpeg",
        lambda font_path: calls.append(font_path) or [],
    )

    pipeline.preflight(backgrounds, font_path=first_font)
    pipeline.preflight(backgrounds, font_path=first_font)
    pipeline.preflight(backgrounds, font_path=second_font)

    assert calls == [first_font.resolve(), second_font.resolve()]


def test_preflight_reports_missing_font(monkeypatch, tmp_path):
    backgrounds, font = configure_preflight(monkeypatch, tmp_path)
    with pytest.raises(PreflightError, match="Font file does not exist"):
        pipeline.preflight(backgrounds, font_path=font.with_name("missing.ttf"))


def test_preflight_reports_missing_and_empty_backgrounds(monkeypatch, tmp_path):
    _, font = configure_preflight(monkeypatch, tmp_path)
    with pytest.raises(PreflightError, match="does not exist"):
        pipeline.preflight(tmp_path / "missing", font_path=font)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PreflightError, match="No video files"):
        pipeline.preflight(empty, font_path=font)


def test_preflight_reports_filter_failure(monkeypatch, tmp_path):
    backgrounds, font = configure_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", "failed"))
    with pytest.raises(PreflightError, match="Unable to inspect ffmpeg filters: failed"):
        pipeline.preflight(backgrounds, font_path=font)


def test_preflight_requires_subtitles_filter(monkeypatch, tmp_path):
    backgrounds, font = configure_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: successful_run(" V->A volume"))
    with pytest.raises(PreflightError, match="required subtitles filter"):
        pipeline.preflight(backgrounds, font_path=font)


def test_cli_returns_one_when_preflight_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(PreflightError("invalid assets")),
    )
    assert cli.main([
        "--input-dir",
        str(tmp_path / "input"),
        "--output-dir",
        str(tmp_path / "output"),
        "--backgrounds-dir",
        str(tmp_path / "backgrounds"),
    ]) == 1


def test_main_shim_propagates_cli_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda: 7)

    with pytest.raises(SystemExit) as error:
        runpy.run_path(Path(__file__).parents[1] / "main.py", run_name="__main__")

    assert error.value.code == 7


def test_worker_logs_processing_and_runtime_on_failure(monkeypatch, tmp_path, caplog):
    events = []
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"

    monkeypatch.setattr(cli, "configure_logging", lambda: events.append("configured"))
    monkeypatch.setattr(cli, "time", SimpleNamespace(time=iter([10, 12.345]).__next__))

    def fail_create_short(*args):
        events.append("create_short")
        raise RuntimeError("failed")

    monkeypatch.setattr(cli, "create_short", fail_create_short)
    with caplog.at_level("INFO", logger=cli.__name__):
        with pytest.raises(RuntimeError, match="failed"):
            cli._worker(input_path, output_path, tmp_path)

    assert events == ["configured", "create_short"]
    assert [record.getMessage() for record in caplog.records] == [
        "Processing: input.mp4",
        "Runtime: 2.35 - input.mp4",
    ]


def test_select_video_codec_override_does_not_probe(monkeypatch):
    monkeypatch.setattr(codec.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected probe"))
    settings = Settings(video_codec="configured")
    assert codec.select_video_codec(settings) == "configured"
    assert codec.select_video_codec(settings) == "configured"


def test_select_video_codec_caches_hardware_candidate(monkeypatch):
    monkeypatch.setattr(codec.sys, "platform", "linux")
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        return successful_run(" V..... h264_nvenc NVIDIA\n") if "-encoders" in command else successful_run()
    monkeypatch.setattr(codec.subprocess, "run", fake_run)
    assert codec.select_video_codec() == "h264_nvenc"
    assert codec.select_video_codec() == "h264_nvenc"
    assert len(commands) == 2
    assert "color=size=64x64:duration=0.1" in commands[1]


def test_select_video_codec_falls_back_and_can_be_invalidated(monkeypatch):
    monkeypatch.setattr(codec.sys, "platform", "darwin")
    monkeypatch.setattr(codec.subprocess, "run", lambda command, **kwargs: successful_run(" V..... h264_videotoolbox VideoToolbox\n"))
    assert codec.select_video_codec() == "h264_videotoolbox"
    codec.invalidate_video_codec("h264_videotoolbox")
    assert codec.select_video_codec() == "libx264"


def test_select_video_codec_tries_each_available_candidate_before_software(monkeypatch):
    monkeypatch.setattr(codec.sys, "platform", "linux")
    tested = []

    def fake_run(command, **kwargs):
        if "-encoders" in command:
            return successful_run(
                " V..... h264_nvenc NVIDIA encoder\n V..... h264_qsv Intel encoder\n"
            )
        tested.append(command[command.index("-c:v") + 1])
        return subprocess.CompletedProcess(command, 1, "", "encode failed")

    monkeypatch.setattr(codec.subprocess, "run", fake_run)

    assert codec.select_video_codec() == "libx264"
    assert tested == ["h264_nvenc", "h264_qsv"]


def patch_create_short(monkeypatch, tmp_path, has_audio=True):
    source = tmp_path / "clip.mp4"
    source.touch()
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    (backgrounds / "bg.mp4").touch()
    font = tmp_path / "font.ttf"
    font.touch()
    monkeypatch.setattr(pipeline, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "resolved_font", lambda settings: as_file(font))
    monkeypatch.setattr(pipeline, "probe_video", lambda path: {"duration": 2, "fps": Fraction(30), "video_stream_index": 0, "has_audio": has_audio})
    monkeypatch.setattr(pipeline, "select_background", lambda *args: (backgrounds / "bg.mp4", 0))
    monkeypatch.setattr(pipeline, "extract_audio", lambda *args: None)
    monkeypatch.setattr(pipeline, "transcribe_words", lambda *args: [])
    monkeypatch.setattr(pipeline, "write_ass_captions", lambda *args: None)
    monkeypatch.setattr(pipeline, "build_filter_complex", lambda *args: "graph")
    monkeypatch.setattr(pipeline, "select_video_codec", lambda settings: "h264_test")
    return source, backgrounds


def test_create_short_raises_no_audio_error(monkeypatch, tmp_path):
    source, backgrounds = patch_create_short(monkeypatch, tmp_path, has_audio=False)
    with pytest.raises(NoAudioError, match="no audio stream"):
        create_short(source, tmp_path / "out.mp4", backgrounds)


def test_create_short_missing_input_raises_probe_error(monkeypatch, tmp_path):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    monkeypatch.setattr(pipeline, "preflight", lambda *args, **kwargs: None)
    with pytest.raises(ProbeError, match="does not exist"):
        create_short(tmp_path / "missing.mp4", tmp_path / "out.mp4", backgrounds)


def test_create_short_output_setup_failure_raises_render_error(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RenderError, match="Unable to prepare output path"):
        create_short(source, blocked_parent / "out.mp4", backgrounds)


def test_create_short_rejects_non_settings_before_path_access():
    with pytest.raises(PreflightError) as error:
        create_short(None, None, None, object())

    assert str(error.value) == "settings must be a Settings instance"


def test_create_short_happy_path_retries_codec_and_returns_path(monkeypatch, tmp_path, caplog):
    source, backgrounds = patch_create_short(monkeypatch, tmp_path)
    codecs = []
    invalidated = []
    def fake_render(*args):
        selected = args[6]
        codecs.append(selected)
        if selected == "libx264":
            Path(args[5]).write_bytes(b"video")
            return successful_run()
        return subprocess.CompletedProcess([], 1, "", "failed")
    monkeypatch.setattr(pipeline, "render_video", fake_render)
    monkeypatch.setattr(pipeline, "invalidate_video_codec", lambda selected, settings: invalidated.append(selected))
    output = tmp_path / "nested" / "out.mp4"
    with caplog.at_level("INFO", logger=pipeline.__name__):
        assert create_short(source, output, backgrounds) == output.resolve()
    assert output.read_bytes() == b"video"
    assert codecs == ["h264_test", "libx264"]
    assert invalidated == ["h264_test"]
    assert "Saving: out.mp4" in [record.getMessage() for record in caplog.records]


def test_suppress_stderr_debug_bypasses_file_descriptor_changes(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(transcriber.os, "dup", lambda descriptor: pytest.fail("unexpected dup"))

    with transcriber._suppress_stderr():
        pass


def test_transcriber_suppresses_model_load_and_recognize(monkeypatch, tmp_path):
    active = []
    calls = []

    @contextmanager
    def tracked_suppression():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    class Model:
        def with_timestamps(self):
            assert active
            calls.append("timestamps")
            return self

        def recognize(self, audio_path):
            assert active
            calls.append(("recognize", audio_path))
            return SimpleNamespace(text="")

    def load_model(name, *, quantization):
        assert active
        calls.append(("load", name, quantization))
        return Model()

    transcriber._MODELS.clear()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(transcriber, "_suppress_stderr", tracked_suppression)
    monkeypatch.setattr(transcriber.onnx_asr, "load_model", load_model)
    audio_path = tmp_path / "audio.wav"

    assert transcriber.transcribe_words(audio_path) == []
    assert calls == [
        ("load", Settings().model_name, Settings().quantization),
        "timestamps",
        ("recognize", str(audio_path)),
    ]
