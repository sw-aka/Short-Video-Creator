#from moviepy.video.io.VideoFileClip import VideoFileClip, VideoClip
from moviepy.video.fx.all import crop as moviepy_crop
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips, ImageClip, CompositeVideoClip
import whisper_timestamped as whisper
from PIL import Image, ImageDraw, ImageFont

import random
import os
import math
import time
import numpy as np

BACKGROUND_VIDEOS_DIR = 'background_videos'

FULL_RESOLUTION = (1080/5, 1920/5)
PERCENT_MAIN_CLIP = 30
TEXT_POSITION_PERCENT = 30
FONT_SIZE = 20

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
        
        pass
    
    def __deinit__(self) -> None:
        if self.clip:
            self.clip.close()
            self.clip = None
        if self.background_clip:
            self.background_clip.close()
            self.background_clip = None
        
    def process(self):
        
        self.clip = self.create_final_clip()
        # self.clip.write_videofile("output.mp4", codec="libx264", audio_codec="aac",)
        transcription = self.create_transcription(self.clip, self.audio)
        self.clip = self.add_captions_to_video(self.clip, transcription)

        self.clip.write_videofile("output.mp4", codec="libx264", audio_codec="aac",)

    def create_final_clip(self):
        self.background_clip = BackgroudVideo.get_clip(self.clip.duration)
        
        _, background_height = self.background_clip.size
        target_dimensions = (FULL_RESOLUTION[0], FULL_RESOLUTION[1] - background_height)
        self.clip = VideoTools(self.clip).crop(target_dimensions[0], target_dimensions[1])

        self.clip = clips_array([[self.clip], [self.background_clip]])
        return self.clip

    def create_transcription(self, clip, audio):

        # audio = clip.audio

        print(type(clip))
        print(type(audio))

        os.makedirs("temp", exist_ok=True)

        file_dir = f"temp/{time.time() * 10**20:.0f}.mp3"
        audio.write_audiofile(file_dir, codec="mp3")

        loaded_audio = whisper.load_audio("audio.mp3")
        model = whisper.load_model("whisper-small.en", device="cpu")
        result = whisper.transcribe(model, loaded_audio, language="en")

        os.remove(file_dir)

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
        for pos, timestamp in enumerate(timestamps):
            
            start, end = timestamp["timestamp"]
            text = timestamp["text"]

            if start > previous_time:
                clips.append(clip.subclip(previous_time, start))

            if pos + 1 < len(timestamps):
                next_timestamp_start = timestamps[pos + 1]['timestamp'][0]
                if next_timestamp_start > end:
                    if next_timestamp_start - end > 0.5:
                        end += 0.5
                    else:
                        end = next_timestamp_start
            
            
            previous_time = end
            
            clips.append(
                self.add_text_to_video(
                    clip.subclip(start, end),
                    text
                )
            )
        
        
        clip = concatenate_videoclips(clips)

        clip = clip.subclip(
            timestamps[0]["timestamp"][0],
            timestamps[-1]["timestamp"][1]
        )

        return clip



    def add_text_to_video(self, clip, text):
        
        text_image = self.create_text_image(
            text,
            "SweetieBubbleGum-Regular.ttf",
            FONT_SIZE,
            clip.size[0]
            )
        
        # text_image.show()
        image_clip = ImageClip(np.array(text_image), duration=clip.duration)

        y_offset = round(FULL_RESOLUTION[1] * (TEXT_POSITION_PERCENT / 100))
        clip = CompositeVideoClip([clip, image_clip.set_position((0, y_offset,))]) #clips_array([[clip], [image_clip]])
        
        return clip

    def create_text_image(self, text, font_path, font_size, max_width):
        # Create a blank image with a white background
        image = Image.new("RGBA", (max_width, 1000), (0,0,0,0))
        draw = ImageDraw.Draw(image)

        # Load the font
        font = ImageFont.truetype(font_path, font_size)

        # Split the text into words
        words = text.split()

        # Initialize variables
        current_line = ""
        y_position = 10

        text_max_width = max_width - 50

        for word in words:
            # Check if adding the next word exceeds the max width
            text_width = draw.textlength(current_line + " " + word, font)
            text_height = font_size
            if text_width <= text_max_width:
                # If not, add the word to the current line
                if current_line:
                    current_line += " "
                current_line += word
            else:
                # Calculate the X-position to center the text
                x_position = (text_max_width - draw.textlength(current_line, font)) // 2
                # Draw the current line at the calculated position
                draw.text((x_position, y_position), current_line, font=font, fill="white", stroke_width=5, stroke_fill='black')
                y_position += text_height
                current_line = word

        # Calculate the X-position for the last line to center the text
        x_position = (max_width - draw.textlength(current_line, font)) // 2
        # Draw the last line at the calculated position
        draw.text((x_position, y_position), current_line, font=font, fill="white", stroke_width=5, stroke_fill='black')

        # Crop the image to the actual content size
        image = image.crop((0, 0, max_width, y_position + text_height + 100))
        # return image
        
        return image

VideoCreation(VideoFileClip('video.mp4')).process()






