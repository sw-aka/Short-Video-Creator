import os
from moviepy.editor import VideoFileClip
import time


INPUT_VIDEOS_DIR = "videos_to_split"
OUTPUT_VIDEOS_DIR = "input_videos"

def find_black_frames(video_path, threshold=10, fps=30):
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

os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)

input_files = os.listdir(INPUT_VIDEOS_DIR)

for file_name in input_files:
    print(f"Loading timestamps for: {file_name}")
    black_frames = find_black_frames(os.path.join(INPUT_VIDEOS_DIR, file_name))

    clip = VideoFileClip(os.path.join(INPUT_VIDEOS_DIR, file_name))

    timestamps_string = ''
    for pos, (start, end) in enumerate(black_frames):
        if pos == 0:
            continue

        if black_frames[pos][0] < 5:
            continue

        print(f"Saving video {pos}/{len(black_frames)-1}")

        if  black_frames[pos][0] - black_frames[pos-1][1] + 0.02 < 5:
            continue

        short_clip = clip.subclip(black_frames[pos-1][1] + 0.02, black_frames[pos][0])

        
        file_dir = f"{OUTPUT_VIDEOS_DIR}/{time.time() * 10**20:.0f}.mp4"

        try:
            short_clip.write_videofile(file_dir, codec="h264_nvenc", audio_codec="aac", fps=clip.fps, threads = 32, verbose=False, logger=None)
        except AttributeError:
            print("Error saving clip.")


        short_clip.close()

    clip.close()
        
