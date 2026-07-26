from pathlib import Path


## Model settings
MODEL_NAME = 'nemo-parakeet-tdt-0.6b-v3'
QUANTIZATION = 'int8'

## Processing settings
MAX_NUMBER_OF_PROCESSES = 1 # The maximum number of videos which can be processed simultaneously
NUM_THREADS = 12 # The number of threads used to save the editted video

## Font settings
FONT_NAME = 'Super Carnival.ttf' # The name of the font file used for captions
FONT_SIZE = 100
FONT_BORDER_WEIGHT = 10

## Video settings
FULL_RESOLUTION = (1080, 1920) # Resolution of the outputted video (width, height) in pixels
PERCENT_MAIN_CLIP = 40 # Percentage of output video height which is the main video (not the background video)
TEXT_POSITION_PERCENT = 30 # Position of caption text as a percentage of video height (from top of video)
VIDEO_CODEC = None # Video encoder override; None automatically selects an available encoder
VIDEO_BITRATE = '8M' # Explicit bitrate used to maintain output video quality

## Source folders
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_VIDEOS_DIR = PROJECT_ROOT / 'INPUT_VIDEOS' # Directory of the input videos
OUTPUT_VIDEOS_DIR = PROJECT_ROOT / 'OUTPUT_VIDEOS' # Directory the editted videos will be saved
BACKGROUND_VIDEOS_DIR = PROJECT_ROOT / 'BACKGROUND_VIDEOS' # Directory of the background videos
FONTS_DIR = PROJECT_ROOT / 'FONTS' # Directory the fonts are stored in
