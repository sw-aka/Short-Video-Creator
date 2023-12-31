from moviepy.video.io.VideoFileClip import VideoFileClip
import pyperclip

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

# Example usage
video_path = "input_videos\example.mp4"
black_frames = find_black_frames(video_path)

# Print the timestamps of black frames

timestamps_string = ''
for pos, (start, end) in enumerate(black_frames):
    if pos == 0:
        continue

    if black_frames[pos][0] < 5:
        continue
    
    timestamps_string += f"{black_frames[pos-1][1] + 0.02 :.2f}:{black_frames[pos][0] :.2f},"
    print(f"Black Frame: {start:.2f}s - {end:.2f}s")
timestamps_string = timestamps_string.rstrip(',')

print(timestamps_string)
pyperclip.copy(timestamps_string)
print('Timestamps copied to clipboard.')