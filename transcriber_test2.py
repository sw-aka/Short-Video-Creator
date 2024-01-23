import whisper_timestamped as whisper
import time
import numpy as np
start_time = time.time()
# audio = whisper.load_audio("audio.mp3")
#TypeError: expected np.ndarray (got AudioFileClip)
from moviepy.editor import VideoFileClip

video = VideoFileClip("video.mp4")

audio_data = list(video.audio.iter_chunks(chunksize=44100))
audio_array = np.vstack(audio_data)


print(type(audio_array))

# exit()

# audio = np.array(video.audio.to_soundarray())

# audio_chunks = list(video.audio.iter_chunks())

# Concatenate the audio chunks into a single array
# audio = np.concatenate(audio_chunks)
video.close()


model = whisper.load_model("whisper-large-v3", device="cpu")

result = whisper.transcribe(model, audio_array, language="en")

import json
print(json.dumps(result, indent = 2, ensure_ascii = False))

print(f"Runtime: {round(time.time()-start_time)}")