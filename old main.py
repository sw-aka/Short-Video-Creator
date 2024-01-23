import os
import time
import secrets
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import crop
import time
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips, VideoClip, ImageSequenceClip, CompositeVideoClip, ImageClip

from moviepy.editor import VideoFileClip, clips_array
import random
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import concurrent.futures
import multiprocessing
import whisper_timestamped as whisper


INPUT_VIDEOS_PATH = 'input_videos'
OUTPUT_VIDEOS_PATH = 'output_videos'
TIMESTAMPS_PATH = 'timestamps.txt'

class Video:

    @staticmethod
    def crop_to_aspect_ratio(clip, aspect_ratio):
        # Calculate the current aspect ratio
        current_aspect_ratio = clip.size[0] / clip.size[1]

        # Calculate the cropping dimensions based on the target aspect ratio
        if current_aspect_ratio > aspect_ratio:
            # Crop horizontally
            new_width = int(aspect_ratio * clip.size[1])
            crop_x = (clip.size[0] - new_width) / 2
            crop_y = 0
        else:
            # Crop vertically
            new_height = int(clip.size[0] / aspect_ratio)
            crop_x = 0
            crop_y = (clip.size[1] - new_height) / 2

        # Apply the cropping
        return crop(clip, x1=crop_x, x2=clip.size[0]-crop_x, y1=crop_y, y2=clip.size[1]-crop_y)

class ProcessMultiClipVideo:

    def __init__(self, input_path, timestamps) -> None:
        self.original_path = input_path

        # original_video = VideoFileClip(self.original_path)

        # for timestamp_pair in timestamps:
        #     ProcessVideo(original_video.subclip(timestamp_pair[0], timestamp_pair[1]))
        
        args_list = []
        for timestamp_pair in timestamps[4:]:
            print(timestamp_pair)
            self.process_video_wrapper((self.original_path, timestamp_pair,))
            # args_list.append((self.original_path, timestamp_pair))

        # Use ThreadPoolExecutor to run the function with a pool of 5 threads
        # with multiprocessing.Pool(processes=2) as pool:
        #     pool.map(self.process_video_wrapper, args_list)

        # pool.close()
        # pool.join()

        # original_video.close()

    def process_video_wrapper(self, args):
        input_dir, timestamp_pair = args

        if not 5 < timestamp_pair[1] - timestamp_pair[0] < 60:
            return
        
        original_video = VideoFileClip(input_dir)
        
        # try:
        ProcessVideo(original_video.subclip(timestamp_pair[0], timestamp_pair[1]))
        # except:
        #     print('An error occured.')
        
        original_video.close()


    


class ProcessVideo:

    CLIP_ASPECT_RATIO = 9 / 6
    CLIP_RESOLUTION = (540, 360)

    PHONE_ASPECT_RATIO = 9 / 16
    BACKGROUND_CLIP_ASPECT_RATIO = 9 / 10
    BACKGROUND_CLIP_RESOLUTION = (540, 600)

    OUTPUT_RESOLUTION = (540, 960)

    def __init__(self, clip):

        self.create_working_directory()

        print(clip.duration)
        clip.write_videofile(os.path.join(self.working_dir, 'input.mp4'), codec="libx264", audio_codec="aac", fps=clip.fps)
        clip.close()

        self.input_dir = os.path.join(self.working_dir, 'input.mp4')

        self.extract_audio()

        clip = self.crop_input_video()
        background_clip = self.get_background_clip(clip.duration)
        print(f"Backgound clip duration: {background_clip.duration}")
        stacked_video = self.stack_clips(clip, background_clip)
        print(f"Stacked video duration: {stacked_video.duration}")
        transcription = self.transcribe_audio()

        self.add_captions_to_video(stacked_video, transcription)

        clip.close()
        background_clip.close()
        stacked_video.close()
        
        self.copy_to_output_and_delete()

        
        
    def create_working_directory(self):
        unique_id = ''.join(secrets.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(10))
        if os.path.exists(f'working_directory/{unique_id}'):
            return self.create_working_directory()
        os.makedirs(f'working_directory/{unique_id}', exist_ok=True)

        self.working_dir = f'working_directory/{unique_id}'
    
    def extract_audio(self):
        self.audio_path = os.path.join(self.working_dir, 'audio.mp3')

        video_clip = VideoFileClip(self.input_dir)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(self.audio_path, codec='mp3')
        
        video_clip.close()
        audio_clip.close()
        

    def crop_input_video(self) -> VideoFileClip:
        # Load the video clip
        video_clip = VideoFileClip(self.input_dir)
        cropped_clip = Video.crop_to_aspect_ratio(video_clip, self.CLIP_ASPECT_RATIO)
        resized_clip = cropped_clip.resize(self.CLIP_RESOLUTION)
        return resized_clip
    
        # Write the output video file
        # cropped_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    def get_background_clip(self, duration) -> VideoFileClip:
    
        background_video = VideoFileClip('background.mp4')
        start_pos = random.randint(0, int(background_video.duration - duration))
        background_clip = background_video.subclip(start_pos, start_pos + duration)
        
        temp_clip = Video.crop_to_aspect_ratio(background_clip, self.PHONE_ASPECT_RATIO)
        cropped_clip = Video.crop_to_aspect_ratio(temp_clip, self.BACKGROUND_CLIP_ASPECT_RATIO)

        resized_clip = cropped_clip.resize(self.BACKGROUND_CLIP_RESOLUTION)
  
        # resized_clip.write_videofile(os.path.join(self.working_dir, 'background_clip.mp4'), codec='libx264', audio_codec='aac')
        return resized_clip.set_audio(None)
    
    def stack_clips(self, clip, background_clip):
        final_clip = clips_array([[clip], [background_clip]])
        return final_clip.resize(self.OUTPUT_RESOLUTION)
    
    def transcribe_audio(self):

        # Replace 'path/to/local/model' with the actual path to the directory containing the model files
        local_model_path = 'whisper-large-v3'

        sample = self.audio_path

        audio = whisper.load_audio(sample)

        model = whisper.load_model(local_model_path, device="cpu")

        result = whisper.transcribe(model, audio, language="en")

        output = []
        for segment in result['segments']:
            for word in segment['words']:
                output.append({
                    'timestamp': (word['start'], word['end']),
                    'text': word['text']
                })


        return output
    
    def add_captions_to_video(self, stacked_video, captions):
        video_clip = stacked_video


        def create_image(text, font_path, font_size, max_width):
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
            image.save(os.path.join(self.working_dir, 'text.png'))

            image.close()

        # Function to add text to a subclip
        def add_text(subclip, txt, fontsize=24, color='white', bg_color='transparent'):

            font_path = "LuckiestGuy.ttf"  # Replace with the path to your font file
            font_size = 40
            max_width = 500

            create_image(txt, font_path, font_size, max_width)

            image_clip = ImageSequenceClip([os.path.join(self.working_dir, 'text.png')], fps=24)  # Adjust fps as needed

            centered_image_clip = image_clip.set_position(("center", "center"))

            # Create the video by repeating the image for the specified duration
            video_clip = centered_image_clip.set_duration(subclip.duration)

            # Combine the subclip and the centered image
            final_clip = CompositeVideoClip([subclip, video_clip])

            image_clip.close()
            centered_image_clip.close()

            return final_clip
        

        caption_clips = []
        last_clip_end = 0
        for item in captions:
            start, end = item['timestamp']
            text = item['text'].strip()

            if start > video_clip.duration or end > video_clip.duration:
                continue

            gap = start - last_clip_end
            if gap > 0:
                caption_clips.append(video_clip.subclip(last_clip_end, start))

            caption_clips.append(add_text(video_clip.subclip(start, end), text))
            last_clip_end = end
        
        if last_clip_end is not None and last_clip_end + 0.1 < video_clip.duration:
            caption_clips.append(video_clip.subclip(last_clip_end, video_clip.duration))
        # Add captions to the video
            
        # Concatenate the caption clips
        if len(caption_clips) > 0:
            final_clip = concatenate_videoclips(caption_clips)
        else:
            final_clip = video_clip
        print(f"Final clip duration: {final_clip.duration}")
        self.output_path = os.path.join(self.working_dir, 'output.mp4')
        # Write the result to a file
        final_clip.write_videofile(self.output_path, codec="libx264", audio_codec="aac", fps=video_clip.fps)

        # Explicitly close the video clip and its processes
        video_clip.close()

    def copy_to_output_and_delete(self):
        # Get the current date and time
        current_datetime = datetime.now()

        # Format the date and time as a string with milliseconds
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S-%f")

        while not os.path.exists(self.output_path):
            time.sleep(0.1)
        shutil.copy(self.output_path, os.path.join(OUTPUT_VIDEOS_PATH, f'video_{formatted_datetime}.mp4'))
        shutil.rmtree(self.working_dir)

def get_input_video_paths():
    if os.path.exists(INPUT_VIDEOS_PATH):
        return [os.path.join(INPUT_VIDEOS_PATH, x) for x in os.listdir(INPUT_VIDEOS_PATH)]
    else:
        return []


class Timestamps:

    def __init__(self):
        
        with open(TIMESTAMPS_PATH, 'w') as f:
            f.write('')

        file = open(TIMESTAMPS_PATH, 'a')

        input_files = os.listdir(INPUT_VIDEOS_PATH)

        for file_name in input_files:
            print(f"Loading timestamps for: {file_name}")
            black_frames = self.find_black_frames(os.path.join(INPUT_VIDEOS_PATH, file_name))

            timestamps_string = ''
            for pos, (start, end) in enumerate(black_frames):
                if pos == 0:
                    continue

                if black_frames[pos][0] < 5:
                    continue
                
                timestamps_string += f"{black_frames[pos-1][1] + 0.02 :.2f}:{black_frames[pos][0] :.2f},"
                
            timestamps_string = timestamps_string.rstrip(',')

            file.write(f"{file_name}\n")
            file.write(f"{timestamps_string}\n")
        
        file.close()
    
    def find_black_frames(self, video_path, threshold=10, fps=30):
        """
        Find black frames in a video and return their timestamps.

        Parameters:
        - video_path: str, path to the video file
        - threshold: int, threshold for considering a frame as black (0-255)
        - fps: int, frames per second of the video

        Returns:
        - black_frames: list of tuples, each tuple contains the start and end timestamps of a black frame
        """

        # Load the video clip
        video_clip = VideoFileClip(video_path)

        # Initialize variables
        black_frames = []
        frame_duration = 1 / fps
        is_black_frame = False
        start_time = 0

        # Iterate through each frame
        for i, frame in enumerate(video_clip.iter_frames(fps=fps, dtype='uint8')):

            if i * frame_duration < 5:
                continue
            
            # Check if the frame is black based on the threshold
            is_black = frame.mean() < threshold

            # If the frame is black and we are not already in a black frame
            if is_black and not is_black_frame:
                is_black_frame = True
                start_time = i * frame_duration
            # If the frame is not black and we are in a black frame
            elif not is_black and is_black_frame:
                is_black_frame = False
                end_time = i * frame_duration
                black_frames.append((start_time, end_time))

        # Check for the last black frame
        if is_black_frame:
            end_time = video_clip.duration
            black_frames.append((start_time, end_time))

        return black_frames



def get_timestamps():

    #Timestamps()

    if os.path.exists(TIMESTAMPS_PATH):
        with open(TIMESTAMPS_PATH, 'r') as f:
            lines = f.readlines()

        timestmaps = {}
        name = ''
        for pos, line in enumerate(lines):
            line = line.strip()
            if pos % 2 == 0:
                name = line
                continue
            else:
                pairs = []
                timestamp_pairs = line.split(',')
                for pair in timestamp_pairs:
                    start, end = pair.split(':')
                    pairs.append((float(start),float(end),))
                timestmaps[os.path.join(INPUT_VIDEOS_PATH, name)] = pairs
        
        return timestmaps
    else:
        return {}


def main():

    input_video_paths = get_input_video_paths()
    timestamps =  get_timestamps()

    

    for input_video_path in input_video_paths:
        timestamp = None
        if input_video_path in timestamps:
            timestamp = timestamps[input_video_path]
            ProcessMultiClipVideo(input_video_path, timestamp)
        
        # ProcessVideo(input_video_path, timestamp)

    # args_list = []
    # for input_video_path in input_video_paths:
    #     timestamp = timestamps.get(input_video_path, None)
    #     args_list.append((input_video_path, timestamp))

    # # Use ThreadPoolExecutor to run the function with a pool of 5 threads
    # with multiprocessing.Pool(processes=5) as pool:
    #     pool.map(process_video_wrapper, args_list)

    # pool.close()
    # pool.join()

if __name__ == '__main__':
    main()



