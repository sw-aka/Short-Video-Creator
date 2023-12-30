captions = [
    {'timestamp': (0.0, 2.0), 'text': " It's no over! I take picture of Ang Lee!"},
    {'timestamp': (2.0, 4.5), 'text': ' Good! He do too many white people movie anyway!'},
    {'timestamp': (8.0, 9.5), 'text': ' You no come back ever!'},
    {'timestamp': (9.5, 11.0), 'text': ' I know like you American!'},
    {'timestamp': (11.0, 13.0), 'text': ' And all you American look alike!'},
    {'timestamp': (13.0, 16.0), 'text': " Oh, we all look alike, do we? Well look who's talking!"}
]

captions = [{'timestamp': (0.0, 0.84), 'text': " It's no over."}, 
    {'timestamp': (0.84, 2.1), 'text': ' I take picture of Ang Lee.'}, 
    {'timestamp': (2.1, 2.52), 'text': ' Good.'}, 
    {'timestamp': (2.52, 4.5), 'text': ' He do too many white people movie anyway.'}, 
    {'timestamp': (8.0, 9.52), 'text': ' You no come back ever.'}, 
    {'timestamp': (9.52, 11.0), 'text': ' I no like you American.'}, 
    {'timestamp': (11.0, 13.0), 'text': ' And all you American look alike.'}, 
    {'timestamp': (13.0, 14.76), 'text': ' Oh, we all look alike, do we?'}, 
    {'timestamp': (14.76, 16.94), 'text': " Well, look who's talking."}
]

from moviepy.editor import VideoFileClip, TextClip, concatenate_videoclips, CompositeVideoClip

def add_captions(input_video, output_video, captions):
    video_clip = VideoFileClip(input_video)

    # Function to add text to a subclip
    def add_text(subclip, txt, fontsize=24, color='white', bg_color='transparent'):
        txt_clip = TextClip(txt, fontsize=fontsize, bg_color=bg_color, font='Discovery.ttf')
        txt_clip = txt_clip.set_pos('center').set_duration(subclip.duration)
        return CompositeVideoClip([subclip, txt_clip])

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

    # Write the result to a file
    final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=video_clip.fps)

    # Explicitly close the video clip and its processes
    video_clip.close()

# Specify your input and output video filenames
input_video_filename = 'input.mp4'
output_video_filename = 'output.mp4'

# Call the function to add captions to the video
add_captions(input_video_filename, output_video_filename, captions)
