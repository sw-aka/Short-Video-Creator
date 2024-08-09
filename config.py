## Model settings
MODEL_NAME = 'whisper-small.en'
LANGUAGE = 'en'

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

## Source folders
INPUT_VIDEOS_DIR = 'INPUT_VIDEOS' # Directory of the input videos
OUTPUT_VIDEOS_DIR = 'OUTPUT_VIDEOS' # Directory the editted videos will be saved
BACKGROUND_VIDEOS_DIR = 'BACKGROUND_VIDEOS' # Directory of the background videos
FONTS_DIR = 'FONTS' # Directory the fonts are stored in
