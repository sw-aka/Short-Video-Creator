import multiprocessing
import os
import time
import math
import random
import numpy as np

from moviepy.video.fx.all import crop as moviepy_crop
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips, ImageClip, CompositeVideoClip, VideoClip
import whisper_timestamped as whisper
from PIL import Image, ImageDraw, ImageFont


MAX_NUMBER_OF_PROCESSES = 2

INPUT_VIDEOS_DIR = 'input_videos'
OUTPUT_VIDEOS_DIR = 'output_videos'

BACKGROUND_VIDEOS_DIR = 'background_videos'
FONTS_DIR = 'fonts'

FULL_RESOLUTION = (1080, 1920)
PERCENT_MAIN_CLIP = 40
TEXT_POSITION_PERCENT = 30

FONT_SIZE = 100
FONT_BORDER_WEIGHT = 10


class VideoTools:
    clip: VideoFileClip = None
    def __init__(self, clip: VideoFileClip) -> None:
        self.clip = clip

    def __deinit__(self) -> None:
        if self.clip:
            self.clip.close()
            self.clip = None

    def crop(self, width: int, height: int) -> VideoFileClip:
        original_width, original_height = self.clip.size

        width_change_ratio = width / original_width
        height_change_ratio = height / original_height

        max_ratio = max(width_change_ratio, height_change_ratio)

        self.clip = self.clip.resize((
            original_width * max_ratio,
            original_height * max_ratio,
        ))

        new_width, new_height = self.clip.size


        if width_change_ratio > height_change_ratio:
            height_change = new_height - height

            new_y1 = round(height_change / 2)
            new_y2 = min(
                new_y1 + height,
                new_height
            )
            self.clip = moviepy_crop(self.clip, y1 = new_y1, y2 = new_y2)
        elif  height_change_ratio > width_change_ratio:
            width_change = new_width - width

            new_x1 = round(width_change / 2)
            new_x2 = min(
                new_x1 + width,
                new_width
            )
            self.clip = moviepy_crop(self.clip, x1 = new_x1, x2 = new_x2)
            self.clip = self.clip.resize((width, height))
        
        return self.clip

class Tools:

    @staticmethod
    def round_down(num: float, decimals: int = 0) -> float:
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

        return cropped_clip.set_audio(None)

        #cropped_clip.write_videofile('background_clip.mp4', codec="libx264", audio_codec="aac",)
        pass

    @staticmethod
    def select_clip() -> str:
        clips = os.listdir(BACKGROUND_VIDEOS_DIR)
        clip = random.choice(clips)
        return os.path.join(BACKGROUND_VIDEOS_DIR, clip)
    
    @staticmethod
    def trim_clip(clip: VideoFileClip, duration: float) -> VideoFileClip:
        if clip.duration < duration:
            raise ValueError(f"Clip duration {clip.duration} is less than duration {duration}")
        
        clip_start_time = Tools.round_down( random.uniform(0, clip.duration - duration) )
        return clip.subclip(clip_start_time, clip_start_time + duration)

    @staticmethod
    def get_target_resolution():
        return (
            FULL_RESOLUTION[0], 
            round(FULL_RESOLUTION[1] * (1 - (PERCENT_MAIN_CLIP / 100)))
        )
    
    @staticmethod
    def format_all_background_clips():
        clips = os.listdir(BACKGROUND_VIDEOS_DIR)
        for clip_name in clips:
            clip = VideoFileClip(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name))
            clip = VideoTools(clip).crop(FULL_RESOLUTION[0], FULL_RESOLUTION[1])

            clip.write_videofile(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name), codec="libx264", audio_codec="aac",)

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


        os.makedirs("temp", exist_ok=True)

        file_dir = f"temp/{time.time() * 10**20:.0f}.mp3"
        audio.write_audiofile(file_dir, codec="mp3", verbose=False, logger=None)
        
        while not os.path.exists(file_dir):
            time.sleep(0.01)

        loaded_audio = whisper.load_audio(file_dir)
        model = whisper.load_model("whisper-small.en", device="cpu")
        result = whisper.transcribe(model, loaded_audio, language="en", verbose=None)

        try:
            os.remove(file_dir)
        except FileNotFoundError:
            pass

        timestamps = []

        for segment in result['segments']:
            for word in segment['words']:
                timestamps.append({
                    'timestamp': (word['start'], word['end']),
                    'text': word['text']
                })

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
                clips.append(clip.subclip(previous_time, start))
                

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
                    clip.subclip(full_start, end),
                    text
                )
            )

            previous_time = end
            full_start = None
        
        
        if clip.duration - end > 0.01:
            clips.append(
                clip.subclip(end, clip.duration)
            )
            
        
        clip = concatenate_videoclips(clips)

        # try:
        #     clip = clip.subclip(
        #         timestamps[0]["timestamp"][0],
        #         timestamps[-1]["timestamp"][1]
        #     )   
        # except IndexError:
        #     pass
            

        return clip



    def add_text_to_video(self, clip, text):
        
        text_image = self.create_text_image(
            text,
            os.path.join(FONTS_DIR, "SweetieBubbleGum-Regular.ttf"),
            FONT_SIZE,
            clip.size[0]
            )
        
        # text_image.show()
        image_clip = ImageClip(np.array(text_image), duration=clip.duration)

        y_offset = round(FULL_RESOLUTION[1] * (TEXT_POSITION_PERCENT / 100))
        clip = CompositeVideoClip([clip, image_clip.set_position((0, y_offset,))]) #clips_array([[clip], [image_clip]])
        
        return clip

    def create_text_image(self, text, font_path, font_size, max_width):
        image = Image.new("RGBA", (max_width, font_size*10), (0,0,0,0))

        font = ImageFont.truetype(font_path, font_size)

        draw = ImageDraw.Draw(image)

        _, _, w, h = draw.textbbox((0,0), text, font=font)

        draw.text(((max_width - w)/2, round(h * 0.2)), text, font=font, fill="white", stroke_width=FONT_BORDER_WEIGHT, stroke_fill='black')

    
        # ImageDraw.Draw(image).text((0,0), text, font=font, fill="white")

        image = image.crop((0, 0, max_width, round(h * 1.6),))

        return image


def start_process(file_name, processes_status_dict, video_queue: multiprocessing.Queue):

    print(f"Processing: {file_name}")
    start_time = time.time()

    process_identifier = multiprocessing.current_process().pid

    processes_status_dict[process_identifier] = False

    input_video = VideoFileClip(os.path.join(INPUT_VIDEOS_DIR, file_name))
    output_video = VideoCreation(input_video).process()
    
    print(f"Saving: {file_name}")

    output_dir = os.path.join(OUTPUT_VIDEOS_DIR, file_name)
    end_time = round( ((output_video.duration * 100 // output_video.fps) * output_video.fps / 100), 2)
    output_video = output_video.subclip(t_end=end_time)

    for pos in range(5):
        try:
            output_video.write_videofile(output_dir, codec="h264_nvenc", audio_codec="aac", fps=output_video.fps, threads = 32, verbose=False, logger=None)
            break
        except IOError:
            time.sleep(1)
    else:
        print(f"ERROR Saving: {file_name}")
    
    input_video.close()
    output_video.close()

    print(f"Runtime: {round(time.time()-start_time, 2)} - {file_name}")
    
    processes_status_dict[process_identifier] = True







if __name__ == '__main__':

    manager = multiprocessing.Manager()

    processes_status_dict = manager.dict()

    video_queue = multiprocessing.Queue()

    processes = {}

    os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

    input_video_names = os.listdir(INPUT_VIDEOS_DIR)

    for name in input_video_names:
        video_queue.put(name)
        # break ## TEMP
    

    num_active_processes = 0
    
    print('STARTED')

    while (video_queue.qsize() != 0) or (len(processes) != 0):
        
        if (num_active_processes < MAX_NUMBER_OF_PROCESSES) and (video_queue.qsize() > 0):
            
            file_name = video_queue.get()

            p = multiprocessing.Process(target=start_process, args=(file_name, processes_status_dict, video_queue))
            p.start()
            processes[p.pid] = p
            num_active_processes += 1

        for pid, complete in processes_status_dict.items():
            if complete:
                processes[pid].join()
                del processes[pid]
                del processes_status_dict[pid]
                num_active_processes -= 1

        pass
    
    print('MAIN PROCESS COMPLETE')
    pass







