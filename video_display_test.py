import cv2
import numpy as np

# Initialize variables
video_file = "video.mp4"
video = cv2.VideoCapture(video_file)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(video.get(cv2.CAP_PROP_FPS))
current_frame = 0
sub_clips = []
text_file = "sub_clips.txt"

# Function to update sub clip timestamps in the text file
def update_text_file():
    with open(text_file, "w") as file:
        for clip in sub_clips:
            file.write(f"{clip[0]} {clip[1]}\n")

# Function to display video with borders around sub clips
def display_video():
    global current_frame
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if current_frame < frame_count:
            frame_copy = frame.copy()

            # Draw green border around sub clips
            for clip in sub_clips:
                if clip[0] <= current_frame <= clip[1]:
                    cv2.rectangle(frame_copy, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 3)

            cv2.imshow("Video", frame_copy)

            key = cv2.waitKey(30)

            if key == 27:  # ESC key to exit
                break
            elif key == 81:  # Left arrow key to move frame backward
                current_frame = max(0, current_frame - 1)
                video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == 83:  # Right arrow key to move frame forward
                current_frame = min(frame_count - 1, current_frame + 1)
                video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == 32:  # Spacebar to mark start/end of sub clip
                if len(sub_clips) % 2 == 0:
                    sub_clips.append([current_frame, -1])
                else:
                    sub_clips[-1][1] = current_frame
                    update_text_file()
        else:
            break

    video.release()
    cv2.destroyAllWindows()

# Start displaying the video
display_video()
