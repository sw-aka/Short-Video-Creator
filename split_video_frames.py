from moviepy.video.io.VideoFileClip import VideoFileClip

# Function to create subclip
def create_subclip(input_video, start_frame, end_frame, output_filename):
    clip = VideoFileClip(input_video)
    subclip = clip.subclip(start_frame / clip.fps, end_frame / clip.fps)
    subclip.write_videofile(output_filename)

# Read frames from frames.txt
with open('frames.txt', 'r') as file:
    frames = [list(map(int, line.strip().split())) for line in file]

input_video_path = "input_video.mp4"
output_directory = "input_videos/"

# Create subclips
for i, (start_frame, end_frame) in enumerate(frames):
    output_filename = f"{output_directory}subclip_{i}.mp4"
    create_subclip(input_video_path, start_frame, end_frame+5, output_filename)
