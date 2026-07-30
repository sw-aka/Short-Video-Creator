# AI Short Video Creator 
Tool to automatically create short clips with a background video and AI-generated captions.
Can be used for YouTube Shorts, TikTok, Instagram Reels, Snapchat Spotlight.

**No API keys** are required, the videos are processed locally on your computer.


## Table of Contents
- [Example](#example-output)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Example Output
<video width="630" height="300" src="https://github.com/user-attachments/assets/f9e787e9-8de8-48da-9303-956cd58a45f0.mp4" title="Example Output"></video>

<img src="https://github.com/user-attachments/assets/86b149d3-55f2-4e74-b51f-b607a781dc22" width="200" title="Example Output - Screenshot 1"/>
<img src="https://github.com/user-attachments/assets/0a59928f-fe80-4da9-946d-06a0ec6d1660" width="200" title="Example Output - Screenshot 2"/>

## Requirements
- Python >=3.11 - [Download Here](https://www.python.org/downloads)
- FFmpeg is bundled through `imageio-ffmpeg`, so a separate installation is not required. A full system [FFmpeg](https://ffmpeg.org/download.html) installation is optional; its `ffprobe` improves media probing when available.

Captions are transcribed locally with NVIDIA's
[Parakeet TDT 0.6B v3](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
model running on [onnx-asr](https://github.com/istupakov/onnx-asr) (works on
Windows, macOS and Linux, CPU-only by default). The model (~600 MB) is
downloaded automatically from Hugging Face on first run and cached locally.


## Installation
1. Clone the repository:
```bash
 git clone https://github.com/sw-aka/Short-Video-Creator.git
```

2. Install the package:
```bash
 pip install -e .
```

## Usage
1. Move main videos (`.mp4`, `.mov`, `.mkv`, `.avi`, or `.webm`) into ```INPUT_VIDEOS```
2. Move background videos in any of those formats into ```BACKGROUND_VIDEOS```
3. Run the CLI:
 ```bash
 svc
 ```
4. The edited videos are saved in ```OUTPUT_VIDEOS```
5. If any video fails to process, the tool prints a failure summary and exits with a nonzero status.

The compatibility command `python main.py` and `python -m short_video_creator`
provide the same folder-batch workflow. Directory defaults are relative to the
current working directory and can be changed with `--input-dir`, `--output-dir`,
`--backgrounds-dir`, and `--processes`.

### Library API

```python
from short_video_creator import Settings, create_short

output = create_short(
    input_video="clip.mp4",
    output_path="out/short.mp4",
    backgrounds_dir="backgrounds",
    settings=Settings(font_size=120),
)
```

`create_short` returns the output `Path`. Catch `ShortVideoError` or one of its
exported subclasses to handle per-clip failures. The default caption font is
bundled; pass `Settings(font_path=Path("font.ttf"))` to use a custom font.


## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes.
4. Push your branch: `git push origin feature-name`.
5. Create a pull request.


## License
This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE.md) License.

Dependencies are installed from PyPI under their own licenses and are not
redistributed here. The Parakeet TDT 0.6B v3 model is released by NVIDIA under
the [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) license.



