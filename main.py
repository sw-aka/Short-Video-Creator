import math
import multiprocessing
import os
import random
import tempfile
import time
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (VideoFileClip, clips_array, concatenate_videoclips,
                     ImageClip, CompositeVideoClip, VideoClip)

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
    VIDEO_CODEC
)

logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'WARNING').upper(), logging.WARNING),
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def list_video_files(directory):
    """Return sorted video file names from a directory."""
    video_extensions = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}
    return sorted(
        name for name in os.listdir(directory)
        if not name.startswith('.')
        and os.path.isfile(os.path.join(directory, name))
        and os.path.splitext(name)[1].lower() in video_extensions
    )


class VideoTools:
    clip: VideoFileClip = None

    def __init__(self, clip: VideoFileClip) -> None:
        self.clip = clip

    def __deinit__(self) -> None:
        if self.clip:
            self.clip.close()
            self.clip = None

    def crop(self, width: int, height: int) -> VideoFileClip:
        """Crop the video clip to the specified width and height.

        Args:
            width (int): The desired width of the cropped video.
            height (int): The desired height of the cropped video.

        Returns:
            VideoFileClip: The cropped video clip.
        """
        original_width, original_height = self.clip.size

        width_change_ratio = width / original_width
        height_change_ratio = height / original_height

        max_ratio = max(width_change_ratio, height_change_ratio)

        self.clip = self.clip.resized((
            original_width * max_ratio,
            original_height * max_ratio,
        ))

        new_width, new_height = self.clip.size

        if width_change_ratio > height_change_ratio:
            height_change = new_height - height
            new_y1 = round(height_change / 2)
            new_y2 = min(new_y1 + height, new_height)
            self.clip = self.clip.cropped(y1=new_y1, y2=new_y2)
        elif height_change_ratio > width_change_ratio:
            width_change = new_width - width
            new_x1 = round(width_change / 2)
            new_x2 = min(new_x1 + width, new_width)
            self.clip = self.clip.cropped(x1=new_x1, x2=new_x2)
            self.clip = self.clip.resized((width, height))

        return self.clip


class Tools:
    @staticmethod
    def round_down(num: float, decimals: int = 0) -> float:
        """
        Rounds down a number to a specified number of decimal places.

        :param num: The number to round down.
        :param decimals: The number of decimal places to round to (default is 0).
        :return: The rounded down number.
        """
        return math.floor(num * 10 ** decimals) / 10 ** decimals

class BackgroudVideo:
    @staticmethod
    def get_clip(duration: float) -> VideoFileClip:
        full_clip = VideoFileClip(BackgroudVideo.select_clip())

        trimmed_clip = BackgroudVideo.trim_clip(full_clip, duration)

        width, height = trimmed_clip.size
        trimmed_clip = VideoTools(trimmed_clip).crop(round(width * 0.9), height)

        target_resolution = BackgroudVideo.get_target_resolution()

        cropped_clip = VideoTools(trimmed_clip).crop(target_resolution[0], target_resolution[1])

        return cropped_clip.without_audio()

    @staticmethod
    def select_clip() -> str:
        clips = list_video_files(BACKGROUND_VIDEOS_DIR)
        clip = random.choice(clips)
        return os.path.join(BACKGROUND_VIDEOS_DIR, clip)

    @staticmethod
    def trim_clip(clip: VideoFileClip, duration: float) -> VideoFileClip:
        """
        Trims a video clip to a specified duration.

        :param clip: The VideoFileClip to trim.
        :param duration: The desired duration of the trimmed clip.
        :return: A trimmed VideoFileClip object.
        :raises ValueError: If the clip's duration is less than the specified duration.
        """
        if clip.duration < duration:
            raise ValueError(f"Clip duration {clip.duration} is less than duration {duration}")

        clip_start_time = Tools.round_down(random.uniform(0, clip.duration - duration))
        return clip.subclipped(clip_start_time, clip_start_time + duration)

    @staticmethod
    def get_target_resolution():
        return (
            FULL_RESOLUTION[0],
            round(FULL_RESOLUTION[1] * (1 - (PERCENT_MAIN_CLIP / 100)))
        )

    @staticmethod
    def format_all_background_clips():
        clips = list_video_files(BACKGROUND_VIDEOS_DIR)
        for clip_name in clips:
            clip = VideoFileClip(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name))
            clip = VideoTools(clip).crop(FULL_RESOLUTION[0], FULL_RESOLUTION[1])

            clip.write_videofile(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name), codec="libx264", audio_codec="aac")

class VideoCreation:
    clip = None
    audio = None
    background_clip = None

    def __init__(self, clip: VideoFileClip) -> None:
        self.clip = clip
        self.audio = clip.audio

    def __deinit__(self) -> None:
        if self.clip:
            self.clip.close()
            self.clip = None
        if self.background_clip:
            self.background_clip.close()
            self.background_clip = None

    def process(self) -> VideoClip:
        self.clip = self.create_final_clip()
        transcription = self.create_transcription(self.audio)
        self.clip = self.add_captions_to_video(self.clip, transcription)

        return self.clip

    def create_final_clip(self):
        self.background_clip = BackgroudVideo.get_clip(self.clip.duration)

        _, background_height = self.background_clip.size
        target_dimensions = (FULL_RESOLUTION[0], FULL_RESOLUTION[1] - background_height)
        self.clip = VideoTools(self.clip).crop(target_dimensions[0], target_dimensions[1])

        self.clip = clips_array([[self.clip], [self.background_clip]])
        return self.clip

    def create_transcription(self, audio):
        fd, file_path = tempfile.mkstemp(prefix="svc-audio-", suffix=".wav")
        os.close(fd)
        try:
            # Save 16 kHz mono WAV audio for ASR
            audio.write_audiofile(file_path, fps=16000, codec="pcm_s16le", ffmpeg_params=["-ac", "1"], logger=None)

            timestamps = transcriber.transcribe_words(file_path)
        finally:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

        return timestamps

    def add_captions_to_video(self, clip, timestamps):
        if len(timestamps) == 0:
            return clip

        clips = []
        previous_time = 0

        queued_texts = []
        full_start = None

        end = 0

        for pos, timestamp in enumerate(timestamps):
            start, end = timestamp["timestamp"]
            text = timestamp["text"]

            if start > previous_time and len(queued_texts) == 0:
                clips.append(clip.subclipped(previous_time, start))

            if pos + 1 < len(timestamps):
                next_timestamp_start = timestamps[pos + 1]['timestamp'][0]
                if next_timestamp_start > end:
                    if next_timestamp_start - end > 0.5:
                        end += 0.5
                    else:
                        end = next_timestamp_start

            if end - previous_time < 0.3 and pos + 1 < len(timestamps):
                if full_start is None:
                    full_start = start
                queued_texts.append(text)
                continue

            queued_texts.append(text)

            if len(queued_texts) > 0:
                text = " ".join(queued_texts)
                queued_texts = []

            if full_start is None:
                full_start = start

            if full_start > clip.duration or end > clip.duration:
                continue

            clips.append(
                self.add_text_to_video(
                    clip.subclipped(full_start, end),
                    text
                )
            )

            previous_time = end
            full_start = None

        if clip.duration - end > 0.01:
            clips.append(
                clip.subclipped(end, clip.duration)
            )

        clip = concatenate_videoclips(clips)

        return clip

    def add_text_to_video(self, clip, text):
        text_image = self.create_text_image(
            text,
            os.path.join(FONTS_DIR, FONT_NAME),
            FONT_SIZE,
            clip.size[0]
        )

        image_clip = ImageClip(np.array(text_image), duration=clip.duration)

        y_offset = round(FULL_RESOLUTION[1] * (TEXT_POSITION_PERCENT / 100))
        clip = CompositeVideoClip([clip, image_clip.with_position((0, y_offset,))])

        return clip

    def create_text_image(self, text, font_path, font_size, max_width):
        image = Image.new("RGBA", (max_width, font_size * 10), (0, 0, 0, 0))

        font = ImageFont.truetype(font_path, font_size)

        draw = ImageDraw.Draw(image)

        _, _, w, h = draw.textbbox((0, 0), text, font=font)

        draw.text(((max_width - w) / 2, round(h * 0.2)), text, font=font, fill="white", stroke_width=FONT_BORDER_WEIGHT, stroke_fill='black')

        image = image.crop((0, 0, max_width, round(h * 1.6),))

        return image


def start_process(file_name, processes_status_dict):
    """
    Process a video file by applying transformations and saving the output.

    Args:
        file_name (str): The name of the video file to process.
        processes_status_dict (dict): A dictionary to track the status of processes.
    """

    logging.info(f"Processing: {file_name}")
    start_time = time.time()

    process_identifier = multiprocessing.current_process().pid

    processes_status_dict[process_identifier] = False

    input_video = VideoFileClip(os.path.join(INPUT_VIDEOS_DIR, file_name))

    output_video = VideoCreation(input_video).process()

    logging.info(f"Saving: {file_name}")

    output_dir = os.path.join(OUTPUT_VIDEOS_DIR, file_name)
    end_time = math.floor(output_video.duration * output_video.fps) / output_video.fps
    if end_time <= 0:
        logging.error(f"ERROR Processing: {file_name}. Frame-aligned duration is nonpositive: {end_time}")
        input_video.close()
        output_video.close()
        processes_status_dict[process_identifier] = True
        return

    output_video = output_video.subclipped(end_time=end_time)

    # Attempt to save the output video, retrying up to 5 times on failure
    video_codec = VIDEO_CODEC
    video_bitrate = VIDEO_BITRATE
    for pos in range(5):
        try:
            output_video.write_videofile(
                output_dir,
                codec=video_codec,
                bitrate=video_bitrate,
                audio_codec="aac",
                fps=output_video.fps,
                threads=NUM_THREADS,
                logger=None
            )
            break
        except Exception as error:
            if video_codec == VIDEO_CODEC:
                video_codec = "libx264"
                video_bitrate = None
            elif not isinstance(error, IOError):
                raise
            logging.warning(f"ERROR Saving: {file_name}. Trying again {pos + 1}/5")
            time.sleep(1)
    else:
        logging.error(f"ERROR Saving: {file_name}")

    input_video.close()
    output_video.close()

    logging.info(f"Runtime: {round(time.time() - start_time, 2)} - {file_name}")

    processes_status_dict[process_identifier] = True


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    processes_status_dict = manager.dict()

    os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

    # List of video files pending processing; only the parent process reads it,
    # so a plain list avoids multiprocessing.Queue's feeder-thread startup race
    pending_videos = list_video_files(INPUT_VIDEOS_DIR)

    processes = {}
    num_active_processes = 0
    logging.info('STARTED')

    while (len(pending_videos) != 0) or (len(processes) != 0):
        if (num_active_processes < MAX_NUMBER_OF_PROCESSES) and (len(pending_videos) != 0):
            file_name = pending_videos.pop(0)

            p = multiprocessing.Process(target=start_process, args=(file_name, processes_status_dict))
            p.start()
            processes[p.pid] = p
            num_active_processes += 1

        for pid, complete in processes_status_dict.items():
            if complete:
                processes[pid].join()
                del processes[pid]
                del processes_status_dict[pid]
                num_active_processes -= 1

    logging.info('MAIN PROCESS COMPLETE')
