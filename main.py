import math
import multiprocessing
import os
import random
import shutil
import time
import logging
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (VideoFileClip, clips_array, concatenate_videoclips,
                             ImageClip, CompositeVideoClip, VideoClip)
from moviepy.video.fx.all import crop as moviepy_crop
import whisper_timestamped as whisper

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
    MODEL_NAME,
    LANGUAGE,
    NUM_THREADS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoTools:
    # Initialize the VideoTools class with a VideoFileClip
    clip: VideoFileClip = None

    def __init__(self, clip: VideoFileClip) -> None:
        """Constructor to initialize the VideoFileClip."""
        self.clip = clip

    def __deinit__(self) -> None:
        """Destructor to clean up resources."""
        if self.clip:
            self.clip.close()  # Close the clip to free resources
            self.clip = None  # Set clip to None to avoid dangling reference

    def crop(self, width: int, height: int) -> VideoFileClip:
        """Crop the video clip to the specified width and height.

        Args:
            width (int): The desired width of the cropped video.
            height (int): The desired height of the cropped video.

        Returns:
            VideoFileClip: The cropped video clip.
        """
        # Get the original dimensions of the video clip
        original_width, original_height = self.clip.size

        # Calculate the change ratios for width and height
        width_change_ratio = width / original_width
        height_change_ratio = height / original_height

        # Determine the maximum ratio to maintain aspect ratio
        max_ratio = max(width_change_ratio, height_change_ratio)

        # Resize the clip based on the maximum ratio
        self.clip = self.clip.resize((
            original_width * max_ratio,
            original_height * max_ratio,
        ))

        # Get the new dimensions after resizing
        new_width, new_height = self.clip.size

        # Crop the video based on the aspect ratio
        if width_change_ratio > height_change_ratio:
            # Calculate the vertical crop
            height_change = new_height - height
            new_y1 = round(height_change / 2)  # Calculate the starting y-coordinate
            new_y2 = min(new_y1 + height, new_height)  # Calculate the ending y-coordinate
            self.clip = moviepy_crop(self.clip, y1=new_y1, y2=new_y2)  # Crop the video
        elif height_change_ratio > width_change_ratio:
            # Calculate the horizontal crop
            width_change = new_width - width
            new_x1 = round(width_change / 2)  # Calculate the starting x-coordinate
            new_x2 = min(new_x1 + width, new_width)  # Calculate the ending x-coordinate
            self.clip = moviepy_crop(self.clip, x1=new_x1, x2=new_x2)  # Crop the video
            self.clip = self.clip.resize((width, height))  # Resize to the final dimensions

        return self.clip  # Return the cropped video clip


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
        """
        Retrieves a random background video clip, trims it to the specified duration,
        and crops it to the target resolution.

        :param duration: The desired duration of the video clip.
        :return: A cropped and trimmed VideoFileClip object.
        """
        # Select a random clip from the background videos directory
        full_clip = VideoFileClip(BackgroudVideo.select_clip())
        
        # Trim the selected clip to the specified duration
        trimmed_clip = BackgroudVideo.trim_clip(full_clip, duration)

        # Crop the trimmed clip to 90% of its width
        width, height = trimmed_clip.size
        trimmed_clip = VideoTools(trimmed_clip).crop(round(width * 0.9), height)

        # Get the target resolution for the final clip
        target_resolution = BackgroudVideo.get_target_resolution()
        
        # Crop the trimmed clip to the target resolution
        cropped_clip = VideoTools(trimmed_clip).crop(target_resolution[0], target_resolution[1])

        # Return the cropped clip without audio
        return cropped_clip.set_audio(None)
    
    @staticmethod
    def select_clip() -> str:
        """
        Selects a random video clip from the background videos directory.

        :return: The file path of the selected video clip.
        """
        clips = os.listdir(BACKGROUND_VIDEOS_DIR)
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
        
        # Randomly select a start time for the subclip
        clip_start_time = Tools.round_down(random.uniform(0, clip.duration - duration))
        return clip.subclip(clip_start_time, clip_start_time + duration)

    @staticmethod
    def get_target_resolution():
        """
        Calculates the target resolution for the video clip based on the full resolution
        and the percentage reduction for the main clip.

        :return: A tuple containing the target width and height.
        """
        return (
            FULL_RESOLUTION[0], 
            round(FULL_RESOLUTION[1] * (1 - (PERCENT_MAIN_CLIP / 100)))
        )
    
    @staticmethod
    def format_all_background_clips():
        """
        Formats all background video clips in the specified directory by cropping them
        to the full resolution and saving them back to the directory.

        :return: None
        """
        clips = os.listdir(BACKGROUND_VIDEOS_DIR)
        for clip_name in clips:
            # Load each clip and crop it to the full resolution
            clip = VideoFileClip(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name))
            clip = VideoTools(clip).crop(FULL_RESOLUTION[0], FULL_RESOLUTION[1])

            # Save the formatted clip back to the directory
            clip.write_videofile(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name), codec="libx264", audio_codec="aac")
            
class VideoCreation:
    # Class attributes for video and audio clips
    clip = None
    audio = None
    background_clip = None

    def __init__(self, clip: VideoFileClip) -> None:
        # Initialize the VideoCreation object with a video clip
        self.clip = clip
        self.audio = clip.audio  # Extract audio from the video clip

    def __deinit__(self) -> None:
        # Clean up resources by closing video and background clips
        if self.clip:
            self.clip.close()
            self.clip = None
        if self.background_clip:
            self.background_clip.close()
            self.background_clip = None

    def process(self) -> VideoClip:
        # Main processing function to create the final video
        self.clip = self.create_final_clip()  # Create the final video clip
        transcription = self.create_transcription(self.audio)  # Generate transcription from audio
        self.clip = self.add_captions_to_video(self.clip, transcription)  # Add captions to the video

        return self.clip  # Return the processed video clip

    def create_final_clip(self):
        # Create the final video clip with a background
        self.background_clip = BackgroudVideo.get_clip(self.clip.duration)  # Get background video clip

        _, background_height = self.background_clip.size  # Get the height of the background clip
        target_dimensions = (FULL_RESOLUTION[0], FULL_RESOLUTION[1] - background_height)  # Calculate target dimensions
        self.clip = VideoTools(self.clip).crop(target_dimensions[0], target_dimensions[1])  # Crop the main clip

        # Combine the main clip and background clip
        self.clip = clips_array([[self.clip], [self.background_clip]])
        return self.clip  # Return the combined clip

    def create_transcription(self, audio):
        # Generate transcription from the audio
        os.makedirs("temp", exist_ok=True)  # Create a temporary directory for audio files

        # Create a unique file name for the audio file
        file_dir = f"temp/{time.time() * 10**20:.0f}.mp3"
        audio.write_audiofile(file_dir, codec="mp3", verbose=False, logger=None)  # Save audio to file

        # Wait until the audio file is created
        while not os.path.exists(file_dir):
            time.sleep(0.01)

        # Load the audio file and transcribe it
        loaded_audio = whisper.load_audio(file_dir)
        model = whisper.load_model(MODEL_NAME, device="cpu")
        result = whisper.transcribe(model, loaded_audio, language=LANGUAGE, verbose=None)

        # Clean up the temporary audio file
        try:
            os.remove(file_dir)
        except FileNotFoundError:
            pass

        timestamps = []  # List to hold timestamps and words

        # Extract timestamps and words from the transcription result
        for segment in result['segments']:
            for word in segment['words']:
                timestamps.append({
                    'timestamp': (word['start'], word['end']),
                    'text': word['text']
                })

        return timestamps  # Return the list of timestamps and words

    def add_captions_to_video(self, clip, timestamps):
        # Add captions to the video based on the provided timestamps
        if len(timestamps) == 0:
            return clip  # Return the original clip if no timestamps

        clips = []  # List to hold video clips with captions
        previous_time = 0  # Track the end time of the previous caption

        queued_texts = []  # List to hold texts for the current caption
        full_start = None  # Start time for the current caption

        end = 0  # End time for the current caption

        # Iterate through the timestamps to create captions
        for pos, timestamp in enumerate(timestamps):
            start, end = timestamp["timestamp"]
            text = timestamp["text"]

            # If there is a gap before the current caption, add the previous clip
            if start > previous_time and len(queued_texts) == 0:
                clips.append(clip.subclip(previous_time, start))

            # Adjust the end time if there is a next timestamp
            if pos + 1 < len(timestamps):
                next_timestamp_start = timestamps[pos + 1]['timestamp'][0]
                if next_timestamp_start > end:
                    if next_timestamp_start - end > 0.5:
                        end += 0.5
                    else:
                        end = next_timestamp_start

            # If the gap between captions is small, queue the text
            if end - previous_time < 0.3 and pos + 1 < len(timestamps):
                if full_start is None:
                    full_start = start
                queued_texts.append(text)
                continue

            queued_texts.append(text)  # Add the current text to the queue

            # Combine queued texts into a single caption
            if len(queued_texts) > 0:
                text = " ".join(queued_texts)
                queued_texts = []

            if full_start is None:
                full_start = start

            # Skip if the caption exceeds the clip duration
            if full_start > clip.duration or end > clip.duration:
                continue

            # Add the captioned clip to the list
            clips.append(
                self.add_text_to_video(
                    clip.subclip(full_start, end),
                    text
                )
            )

            previous_time = end  # Update the previous time
            full_start = None  # Reset full start for the next caption

        # Add any remaining clip after the last caption
        if clip.duration - end > 0.01:
            clips.append(
                clip.subclip(end, clip.duration)
            )

        clip = concatenate_videoclips(clips)  # Concatenate all clips with captions

        return clip  # Return the final clip with captions

    def add_text_to_video(self, clip, text):
        # Add text overlay to the video clip
        text_image = self.create_text_image(
            text,
            os.path.join(FONTS_DIR, FONT_NAME),
            FONT_SIZE,
            clip.size[0]
        )

        image_clip = ImageClip(np.array(text_image), duration=clip.duration)  # Create an image clip for the text

        y_offset = round(FULL_RESOLUTION[1] * (TEXT_POSITION_PERCENT / 100))  # Calculate vertical position for text
        clip = CompositeVideoClip([clip, image_clip.set_position((0, y_offset,))])  # Overlay text on the video

        return clip  # Return the video clip with text

    def create_text_image(self, text, font_path, font_size, max_width):
        # Create an image with the specified text
        image = Image.new("RGBA", (max_width, font_size * 10), (0, 0, 0, 0))  # Create a transparent image

        font = ImageFont.truetype(font_path, font_size)  # Load the specified font

        draw = ImageDraw.Draw(image)  # Create a drawing context

        # Get the bounding box for the text
        _, _, w, h = draw.textbbox((0, 0), text, font=font)

        # Draw the text on the image with stroke for better visibility
        draw.text(((max_width - w) / 2, round(h * 0.2)), text, font=font, fill="white", stroke_width=FONT_BORDER_WEIGHT, stroke_fill='black')

        image = image.crop((0, 0, max_width, round(h * 1.6),))  # Crop the image to the desired size

        return image  # Return the created text image


import os
import time
import shutil
import multiprocessing
from moviepy.editor import VideoFileClip

# Constants for input and output directories
INPUT_VIDEOS_DIR = 'input_videos'
OUTPUT_VIDEOS_DIR = 'output_videos'

def start_process(file_name, processes_status_dict, video_queue: multiprocessing.Queue):
    """
    Process a video file by applying transformations and saving the output.

    Args:
        file_name (str): The name of the video file to process.
        processes_status_dict (dict): A dictionary to track the status of processes.
        video_queue (multiprocessing.Queue): A queue to manage video processing tasks.
    """
    
    logging.info(f"Processing: {file_name}")  # Log the start of processing
    start_time = time.time()  # Record the start time

    # Get the current process identifier
    process_identifier = multiprocessing.current_process().pid

    # Mark the process as not finished in the status dictionary
    processes_status_dict[process_identifier] = False

    # Load the input video file
    input_video = VideoFileClip(os.path.join(INPUT_VIDEOS_DIR, file_name))
    
    # Process the video using a custom VideoCreation class
    output_video = VideoCreation(input_video).process()
    
    logging.info(f"Saving: {file_name}")  # Log the saving process

    # Define the output directory and calculate the end time for the subclip
    output_dir = os.path.join(OUTPUT_VIDEOS_DIR, file_name)
    end_time = round(((output_video.duration * 100 // output_video.fps) * output_video.fps / 100), 2)
    
    # Create a subclip of the output video
    output_video = output_video.subclip(t_end=end_time)

    # Attempt to save the output video, retrying up to 5 times on failure
    for pos in range(5):
        try:
            output_video.write_videofile(
                output_dir,
                codec="libx264",
                audio_codec="aac",
                fps=output_video.fps,
                threads=NUM_THREADS,
                verbose=False,
                logger=None
            )
            break  # Exit the loop if saving is successful
        except IOError:
            logging.warning(f"ERROR Saving: {file_name}. Trying again {pos + 1}/5")  # Log the error and retry
            time.sleep(1)  # Wait before retrying
    else:
        logging.error(f"ERROR Saving: {file_name}")  # Log if all attempts failed
    
    # Close the input and output video files to free resources
    input_video.close()
    output_video.close()

    # Log the runtime of the processing
    logging.info(f"Runtime: {round(time.time() - start_time, 2)} - {file_name}")
    
    # Mark the process as finished in the status dictionary
    processes_status_dict[process_identifier] = True


def delete_temp_folder():
    """
    Delete the temporary folder used for processing videos.
    """
    try:
        shutil.rmtree('temp')  # Remove the 'temp' directory and all its contents
    except (PermissionError, FileNotFoundError):
        pass  # Ignore permission errors if the folder cannot be deleted or is not found


import subprocess

def check_command(command):
    try:
        # Run the command and check if it is installed
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception as e:
        return str(e)

def clone_respository():
    
    
    # Check for Git
    git_version = check_command(['git', '--version'])
    if not git_version:
        raise Exception("Git is not installed. Git must be installed to download model.")

    git_lfs_version = check_command(['git', 'lfs', 'version'])
    if not git_lfs_version:
        raise Exception("Git LFS is not installed. LFS is required to download model. Install Git LFS and try again.")
        

    repo_url = f'https://huggingface.co/openai/{MODEL_NAME}'
    
    logging.info(f"Cloning {repo_url}")
    # Run the git clone command
    subprocess.run(['git', 'clone', repo_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    logging.info(f"Cloned {repo_url}")
    


if __name__ == '__main__':
    # Clean up any temporary folders before starting
    delete_temp_folder()
    
    if not os.path.exists(MODEL_NAME):
        logging.warning(f'Model {MODEL_NAME} not found.')
        logging.info('Downloading model...')
        clone_respository()
        
    # Create a manager for shared data between processes
    manager = multiprocessing.Manager()
    processes_status_dict = manager.dict()  # Dictionary to track process statuses
    video_queue = multiprocessing.Queue()    # Queue to hold video file names

    # Create input and output directories if they don't exist
    os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

    # List all video files in the input directory
    input_video_names = os.listdir(INPUT_VIDEOS_DIR)

    # Add video file names to the queue
    for name in input_video_names:
        video_queue.put(name)

    processes = {} # Dictionary to store processes
    num_active_processes = 0  # Counter for active processes
    logging.info('STARTED')

    # Main loop to manage video processing
    while (video_queue.qsize() != 0) or (len(processes) != 0):
        # Check if we can start a new process
        if (num_active_processes < MAX_NUMBER_OF_PROCESSES) and (video_queue.qsize() > 0):
            file_name = video_queue.get()  # Get the next video file name from the queue

            # Create a new process for video processing
            p = multiprocessing.Process(target=start_process, args=(file_name, processes_status_dict, video_queue))
            p.start()  # Start the process
            processes[p.pid] = p  # Store the process in the dictionary
            num_active_processes += 1  # Increment the active process counter

        # Check for completed processes
        for pid, complete in processes_status_dict.items():
            if complete:  # If the process is complete
                processes[pid].join()  # Wait for the process to finish
                del processes[pid]  # Remove the process from the dictionary
                del processes_status_dict[pid]  # Remove the status from the dictionary
                num_active_processes -= 1  # Decrement the active process counter

    # Clean up temporary folders after processing is complete
    delete_temp_folder()
    logging.info('MAIN PROCESS COMPLETE')






