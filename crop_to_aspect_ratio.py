from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import crop

import time
def crop_video(input_path, output_path, target_aspect_ratio):
    # Load the video clip
    video_clip = VideoFileClip(input_path)

    # Calculate the current aspect ratio
    current_aspect_ratio = video_clip.size[0] / video_clip.size[1]

    # Calculate the cropping dimensions based on the target aspect ratio
    if current_aspect_ratio > target_aspect_ratio:
        # Crop horizontally
        new_width = int(target_aspect_ratio * video_clip.size[1])
        crop_x = (video_clip.size[0] - new_width) / 2
        crop_y = 0
    else:
        # Crop vertically
        new_height = int(video_clip.size[0] / target_aspect_ratio)
        crop_x = 0
        crop_y = (video_clip.size[1] - new_height) / 2

    # Apply the cropping
    cropped_clip = crop(video_clip, x1=crop_x, x2=video_clip.size[0]-crop_x, y1=crop_y, y2=video_clip.size[1]-crop_y)

    # Write the output video file
    cropped_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')


# Example usage
input_video_path = "video.mp4"
output_video_path = "cropped_video.mp4"
phone_aspect_ratio = 19.5 / 9
aspect_ratio = phone_aspect_ratio / 2 # For example, a 16:9 aspect ratio

crop_video(input_video_path, output_video_path, aspect_ratio)


