import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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

# Rest of your code remains unchanged

import time
start_time = time.time()
sample = 'audio.mp3'
result = pipe(sample)
print(result["text"])
print(result)
print(round(time.time()-start_time,2))