import whisper_timestamped as whisper
import time
start_time = time.time()
audio = whisper.load_audio("audio.mp3")

model = whisper.load_model("whisper-large-v3", device="cpu")

result = whisper.transcribe(model, audio, language="en")

import json
print(json.dumps(result, indent = 2, ensure_ascii = False))

print(f"Runtime: {round(time.time()-start_time)}")