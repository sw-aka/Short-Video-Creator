from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips

from moviepy.editor import VideoFileClip, clips_array

def stack_videos(video_path1, video_path2, output_path):
    # Load video clips
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)

    # Stack videos vertically
    final_clip = clips_array([[clip1], [clip2]])

    # Write the final video to the output path
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


def add_audio_to_video(video_path, audio_path, output_path):
    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Load the audio clip
    audio_clip = AudioFileClip(audio_path)

    # Set the audio of the video clip to the loaded audio clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the result to a new video file
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Close the clips
    video_clip.close()
    audio_clip.close()

if __name__ == "__main__":
    video1_path = "cropped_video.mp4"
    video2_path = "cropped_video.mp4"
    output_path = "stacked_video.mp4"

    stack_videos(video1_path, video2_path, output_path)
    # add_audio_to_video(output_path, video1_path, output_path)
