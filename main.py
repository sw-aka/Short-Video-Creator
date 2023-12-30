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




INPUT_VIDEOS_PATH = 'input_videos'
OUTPUT_VIDEOS_PATH = 'output_videos'

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

class ProcessVideo:

    CLIP_ASPECT_RATIO = 9 / 6
    CLIP_RESOLUTION = (540, 360)

    PHONE_ASPECT_RATIO = 9 / 16
    BACKGROUND_CLIP_ASPECT_RATIO = 9 / 10
    BACKGROUND_CLIP_RESOLUTION = (540, 600)

    OUTPUT_RESOLUTION = (540, 960)

    def __init__(self, input_path):
        self.original_path = input_path
        self.create_working_directory()
        self.copy_video_to_working_dir()

        self.extract_audio()

        clip = self.crop_input_video()
        background_clip = self.get_background_clip(clip.duration)

        stacked_video = self.stack_clips(clip, background_clip)

        transcription = self.transcribe_audio()

        self.add_captions_to_video(stacked_video, transcription)

        self.copy_to_output_and_delete()

        
        
    def create_working_directory(self):
        unique_id = ''.join(secrets.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(10))
        if os.path.exists(f'working_directory/{unique_id}'):
            return self.create_working_directory()
        os.makedirs(f'working_directory/{unique_id}', exist_ok=True)

        self.working_dir = f'working_directory/{unique_id}'
    
    def copy_video_to_working_dir(self):
        shutil.copy(self.original_path, os.path.join(self.working_dir, 'input.mp4'))
        self.input_dir = os.path.join(self.working_dir, 'input.mp4')
    
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
    
    def get_background_clip(self, duration):
    
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
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Replace 'path/to/local/model' with the actual path to the directory containing the model files
        local_model_path = 'whisper-large-v3'

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            local_model_path,  # Provide the path to the local directory
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(local_model_path)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

        sample = self.audio_path
        result = pipe(sample)


        return result['chunks']
    
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

            text_max_width = max_width - 10

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
            image.save('text.png')

        # Function to add text to a subclip
        def add_text(subclip, txt, fontsize=24, color='white', bg_color='transparent'):

            font_path = "LuckiestGuy.ttf"  # Replace with the path to your font file
            font_size = 40
            max_width = 500

            create_image(txt, font_path, font_size, max_width)

            image_clip = ImageSequenceClip(['text.png'], fps=24)  # Adjust fps as needed

            centered_image_clip = image_clip.set_position(("center", "center"))

            # Create the video by repeating the image for the specified duration
            video_clip = centered_image_clip.set_duration(subclip.duration)

            # Combine the subclip and the centered image
            final_clip = CompositeVideoClip([subclip, video_clip])
            return final_clip
        

        # Function to add captions to the video
        def add_captions_to_video(clip, caption):
            start, end = caption['timestamp']
            caption_text = caption['text']
            subclip = clip.subclip(start, end)
            return add_text(subclip, caption_text)

        # Add captions to the video
        caption_clips = [add_captions_to_video(video_clip, caption) for caption in captions]
        
        # Concatenate the caption clips
        final_clip = concatenate_videoclips(caption_clips)

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
        # os.remove(self.original_path)

def get_input_video_paths():
    if os.path.exists(INPUT_VIDEOS_PATH):
        return [os.path.join(INPUT_VIDEOS_PATH, x) for x in os.listdir(INPUT_VIDEOS_PATH)]
    else:
        return []


def main():

    input_video_paths = get_input_video_paths()

    for input_video_path in input_video_paths:
        ProcessVideo(input_video_path)


    pass

if __name__ == '__main__':
    main()



