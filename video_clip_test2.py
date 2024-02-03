import tkinter as tk
from PIL import ImageTk, Image
import cv2


def get_total_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    cap.release()

    return total_frames

def get_frame(video, frame_index):
    # Open the video file
    cap = video

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Set the frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame at the specified index
    ret, frame = cap.read()

    # Release the video capture object

    if ret:
        return frame
    else:
        print(f"Error: Could not read frame {frame_index}.")
        return None




def update_image():
    global image_index, tk_images
    image_index = (image_index + 1) % len(images)
    image = images[image_index]
    tk_image = ImageTk.PhotoImage(image)
    image_label.configure(image=tk_image)
    image_label.image = tk_image  # Update reference to avoid garbage collection
    if key_held:
        root.after(100, update_image)  # Schedule the next update after 100 milliseconds

def on_key_press(event):
    global key_held
    if not key_held:
        key_held = True
        update_image()

def on_key_release(event):
    global key_held
    key_held = False



video = cv2.VideoCapture('video.mp4')
number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


print(number_of_frames)

exit()




# Create the main window
root = tk.Tk()
root.title("Image Display")

# Load the images
image_paths = ["image1.jpg", "image2.jpg"]  # Replace with your image file paths
images = [Image.open(path) for path in image_paths]

# Convert the images to a format that Tkinter can display
tk_images = [ImageTk.PhotoImage(image) for image in images]

# Create a label widget to display the image
image_index = 0
image_label = tk.Label(root, image=tk_images[image_index])
image_label.pack()

# Bind the key events
key_held = False
root.bind("<KeyPress>", on_key_press)
root.bind("<KeyRelease>", on_key_release)

# Start the Tkinter event loop
root.mainloop()
