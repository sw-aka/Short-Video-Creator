from moviepy.editor import VideoFileClip

def convert_video_to_mp3(video_path, output_path):
    try:
        # Load the video clip
        video_clip = VideoFileClip(video_path)
        
        # Extract audio from the video
        audio_clip = video_clip.audio
        
        # Save the audio as an MP3 file
        audio_clip.write_audiofile(output_path, codec='mp3')
        
        # Close the video and audio clips
        video_clip.close()
        audio_clip.close()
        
        print(f"Conversion successful. MP3 file saved at: {output_path}")

    except Exception as e:
        print(f"Error during conversion: {e}")

# Example usage
video_file_path = "video.mp4"
output_mp3_path = "audio.mp3"

convert_video_to_mp3(video_file_path, output_mp3_path)
