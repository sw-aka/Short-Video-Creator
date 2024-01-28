from PIL import Image, ImageDraw, ImageFont

image = 

def create_image_with_text(text, font_path, output_path):
    # Set the image size and background color
    width, height = 500, 200
    background_color = (255, 255, 255)  # White

    # Create a new image with a white background
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Load the font
    font_size = 40
    font = ImageFont.truetype(font_path, font_size)

    # Calculate text position
    text_width, text_height = font.getsize(text)
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Draw the text with stroke (outline)
    outline_color = (0, 0, 0)  # Black
    stroke_width = 2

    draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=stroke_width, stroke_fill=outline_color)

    # Save the image
    image.save(output_path)

if __name__ == "__main__":
    input_text = input("Enter the text: ")
    font_path = "SweetieBubbleGum-Regular.ttf"
    output_path = "output_image.png"

    create_image_with_text(input_text, font_path, output_path)
    print(f"Image saved to {output_path}")
