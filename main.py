#from moviepy.video.io.VideoFileClip import VideoFileClip, VideoClip
from moviepy.video.fx.all import crop as moviepy_crop
from moviepy.editor import VideoFileClip, clips_array

import random
import os
import math

BACKGROUND_VIDEOS_DIR = 'background_videos'

FULL_RESOLUTION = (1080/5, 1920/5)
PERCENT_MAIN_CLIP = 35

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
    background_clip = None

    def __init__(self, clip: VideoFileClip) -> None:
        self.clip = clip
        
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

        self.clip.write_videofile("output.mp4", codec="libx264", audio_codec="aac",)

    def create_final_clip(self):
        self.background_clip = BackgroudVideo.get_clip(self.clip.duration)
        
        _, background_height = self.background_clip.size
        target_dimensions = (FULL_RESOLUTION[0], FULL_RESOLUTION[1] - background_height)
        self.clip = VideoTools(self.clip).crop(target_dimensions[0], target_dimensions[1])

        self.clip = clips_array([[self.clip], [self.background_clip]])
        return self.clip

VideoCreation(VideoFileClip('video.mp4')).process()






