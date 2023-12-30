from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import crop
import time
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips, VideoClip, ImageSequenceClip, CompositeVideoClip, ImageClip

from moviepy.editor import VideoFileClip, clips_array

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



def stack_videos(video_path1, video_path2, output_path):
    # Load video clips
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)

    # Stack videos vertically
    final_clip = clips_array([[clip1], [clip2]])

    # Write the final video to the output path
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")



import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def transcribe(audio_path):
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

    sample = audio_path
    result = pipe(sample)


    return result['chunks']


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



from PIL import Image, ImageDraw, ImageFont

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

    for word in words:
        # Check if adding the next word exceeds the max width
        text_width = draw.textlength(current_line + " " + word, font)
        text_height = font_size
        if text_width <= max_width:
            # If not, add the word to the current line
            if current_line:
                current_line += " "
            current_line += word
        else:
            # Calculate the X-position to center the text
            x_position = (max_width - draw.textlength(current_line, font)) // 2
            # Draw the current line at the calculated position
            draw.text((x_position, y_position), current_line, font=font, fill="white", stroke_width=5, stroke_fill='black')
            y_position += text_height
            current_line = word

    # Calculate the X-position for the last line to center the text
    x_position = (max_width - draw.textlength(current_line, font)) // 2
    # Draw the last line at the calculated position
    draw.text((x_position, y_position), current_line, font=font, fill="white", stroke_width=5, stroke_fill='black')

    # Crop the image to the actual content size
    image = image.crop((0, 0, max_width, y_position + text_height))
    # return image
    image.save('text.png')
    # image.show()


from moviepy.editor import VideoFileClip, TextClip, concatenate_videoclips, CompositeVideoClip

def add_captions(input_video, output_video, captions):
    video_clip = VideoFileClip(input_video)

    # Function to add text to a subclip
    def add_text(subclip, txt, fontsize=24, color='white', bg_color='transparent'):

        font_path = "LuckiestGuy.ttf"  # Replace with the path to your font file
        font_size = 70
        max_width = 700

        create_image(txt, font_path, font_size, max_width)

        image_clip = ImageSequenceClip(['text.png'], fps=24)  # Adjust fps as needed

        centered_image_clip = image_clip.set_position(("center", "center"))

        # Create the video by repeating the image for the specified duration
        video_clip = centered_image_clip.set_duration(subclip.duration)

        # Combine the subclip and the centered image
        final_clip = CompositeVideoClip([subclip, video_clip])
        return final_clip
        # # Step 4: Create the video by repeating the image for the specified duration
        # video_clip = image_clip.set_duration(subclip.duration)
        return CompositeVideoClip([subclip.set_position('center'), video_clip])
    

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


if __name__ == "__main__":
    
    input_video_path = "video.mp4"
    output_video_path = "cropped_video.mp4"
    phone_aspect_ratio = 19.5 / 9
    aspect_ratio = phone_aspect_ratio / 2 # For example, a 16:9 aspect ratio

    # crop_video(input_video_path, output_video_path, aspect_ratio)

    # video1_path = "cropped_video.mp4"
    # video2_path = "cropped_video.mp4"
    # output_path = "stacked_video.mp4"
    

    # stack_videos(video1_path, video2_path, output_path)

    # convert_video_to_mp3(input_video_path, "audio.mp3")
    
    # transcription = transcribe('audio.mp3')

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

    add_captions('stacked_video.mp4', 'output.mp4', captions)


