from PIL import Image, ImageDraw, ImageFont

def create_image(text, font_path, font_size, max_width):
    # Create a blank image with a white background
    image = Image.new("RGB", (max_width, 1000), "white")
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Split the text into words
    words = text.split()

    # Initialize variables
    current_line = ""
    y_position = 0

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
            draw.text((x_position, y_position), current_line, font=font, fill="black")
            y_position += text_height
            current_line = word

    # Calculate the X-position for the last line to center the text
    x_position = (max_width - draw.textlength(current_line, font)) // 2
    # Draw the last line at the calculated position
    draw.text((x_position, y_position), current_line, font=font, fill="black")

    # Crop the image to the actual content size
    image = image.crop((0, 0, max_width, y_position + text_height))

    # Save or display the image
    image.save("output.png")
    # image.show()

# Example usage
text = "This is an example text that should be wrapped to fit within a specified max width."
font_path = "LuckiestGuy.ttf"  # Replace with the path to your font file
font_size = 50
max_width = 900

create_image(text, font_path, font_size, max_width)
