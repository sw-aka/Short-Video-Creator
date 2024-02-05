import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time


NUM_EACH_CLICK = 10

right_pressed = False
left_pressed = False

border_colour = "#FFFFFF"
ACTIVE_COLOUR = "green"


root = None
image_label = None

current_frame = 0

clip_beginings = []
clip_ends = []


cap = cv2.VideoCapture('input_video.mp4')

num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_frame(index = 0):
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    # Read the frame at the specified index
    ret, frame = cap.read()

    return frame


def cv2_to_pil_image(cv2_image):
    """Convert a cv2 image to a PIL Image."""
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image

def clip_position(event):

    if current_frame in clip_beginings:
        clip_beginings.remove(current_frame)
        root.configure(bg=border_colour, highlightcolor=border_colour)
    elif current_frame in clip_ends:
        clip_ends.remove(current_frame)
        root.configure(bg=border_colour, highlightcolor=border_colour)
    else:
        if len(clip_beginings) > len(clip_ends):
            clip_ends.append(current_frame)
            root.configure(bg=border_colour, highlightcolor=border_colour)

        else:
            clip_beginings.append(current_frame)
            root.configure(bg=ACTIVE_COLOUR, highlightcolor=ACTIVE_COLOUR)
    
    with open('frames.txt', 'a+') as f:
        f.truncate(0)

        for i, start in enumerate(clip_beginings):
            end = ""
            if i < len(clip_ends):
                end = clip_ends[i]

            f.write(f"{start} {end}\n")

    print(current_frame)

def update_frame_next():
    global current_frame

    if right_pressed:
        if current_frame < num_of_frames - 1 - NUM_EACH_CLICK:
            current_frame += NUM_EACH_CLICK

            if current_frame > num_of_frames - 1:
                current_frame = num_of_frames - 1

            image = ImageTk.PhotoImage(
                cv2_to_pil_image(
                    get_frame(current_frame)
                )
            )
            image_label.configure(
                image = image
            )
            image_label.image = image

            
            if clip_beginings > clip_ends:
                root.configure(bg=ACTIVE_COLOUR, highlightcolor=ACTIVE_COLOUR)
            else:
                root.configure(bg=border_colour, highlightcolor=border_colour)

                if len(clip_ends) != 0:
                    if current_frame < clip_ends[-1]:
                        root.configure(bg=ACTIVE_COLOUR, highlightcolor=ACTIVE_COLOUR)

            
            
            root.after(1, update_frame_next)
        
def update_frame_prev():
    global current_frame

    if left_pressed:
        if current_frame > NUM_EACH_CLICK:
            current_frame -= NUM_EACH_CLICK

            if current_frame < 0:
                current_frame = 0

            image = ImageTk.PhotoImage(
                cv2_to_pil_image(
                    get_frame(current_frame)
                )
            )
            image_label.configure(
                image = image
            )
            image_label.image = image

            if clip_beginings > clip_ends:
                root.configure(bg=ACTIVE_COLOUR, highlightcolor=ACTIVE_COLOUR)
            else:
                root.configure(bg=border_colour, highlightcolor=border_colour)

                if len(clip_ends) != 0:
                    if current_frame < clip_ends[-1]:
                        root.configure(bg=ACTIVE_COLOUR, highlightcolor=ACTIVE_COLOUR)
            
            root.after(1, update_frame_prev)
         

def on_right_key(event):
    global right_pressed
    
    if right_pressed:
        return
    else:
        right_pressed = True
        
        update_frame_next()


def on_right_key_release(event):
    global right_pressed
    right_pressed = False
    pass


def on_left_key(event):
    global left_pressed
    
    if left_pressed:
        return
    else:
        left_pressed = True
        
        update_frame_prev()


def on_left_key_release(event):
    global left_pressed
    left_pressed = False
    pass



def main():
    global right_pressed, root, image_label

    # Create the main window
    root = tk.Tk()
    root.title("Colored Border with Image")

    # Set window size and position
    window_width = 400
    window_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = (screen_width / 2) - (window_width / 2)
    y_coordinate = (screen_height / 2) - (window_height / 2)
    root.geometry("%dx%d+%d+%d" % (window_width, window_height, x_coordinate, y_coordinate))

    # Add a colored border
    border_width = 10  # You can change this width
    root.configure(bg=border_colour, highlightthickness=border_width, highlightcolor=border_colour)

    # Load and display image
    image_path = "image.jpg"  # Path to your image
    # image = Image.open(image_path)
    image = ImageTk.PhotoImage(cv2_to_pil_image(get_frame()))
    image_label = tk.Label(root, image=image, bg=border_colour)
    image_label.pack(expand=True)

    root.bind("<KeyPress-Right>", on_right_key)
    root.bind("<KeyRelease-Right>", on_right_key_release)

    root.bind("<KeyPress-Left>", on_left_key)
    root.bind("<KeyRelease-Left>", on_left_key_release)

    root.bind("<space>", clip_position)


    # Run the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()
