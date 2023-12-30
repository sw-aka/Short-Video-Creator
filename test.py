from PIL import Image, ImageDraw, ImageFont

def draw_text_with_outline(draw, text, position, font, text_color, outline_color, outline_width):
    # Draw the outlined text
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            draw.text((position[0] + dx, position[1] + dy), text, font=font, fill=outline_color)

    # Draw the main text
    draw.text(position, text, font=font, fill=text_color)

def create_image_with_text(text, font_size, max_width):
    # Create a blank image with a white background
    image = Image.new('RGB', (max_width, 100), 'white')
    draw = ImageDraw.Draw(image)

    # Use a truetype font file (replace 'arial.ttf' with your font file)
    font = ImageFont.truetype('arial.ttf', font_size)

    # Initialize variables for text drawing
    current_width = 0
    current_height = 0

    # Set text and outline colors and width
    text_color = 'black'
    outline_color = 'white'
    outline_width = 2

    # Iterate through each word in the text
    for word in text.split():
        # Calculate the width of the word with the current font size
        word_width, _ = draw.textsize(word, font=font)

        # Check if adding the word exceeds the maximum width
        if current_width + word_width > max_width:
            # Start a new line if the maximum width is exceeded
            current_width = 0
            current_height += font.getsize(word)[1]

        # Draw the word on the image
        draw_text_with_outline(draw, word, (current_width, current_height), font, text_color, outline_color, outline_width)

        # Update the current width for the next word
        current_width += word_width + font.getsize(' ')[0]

    return image

if __name__ == "__main__":
    text = "This is a sample text for testing line breaks in the image."
    font_size = 20
    max_width = 300

    image = create_image_with_text(text, font_size, max_width)
    image.show()

