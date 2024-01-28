from PIL import Image, ImageDraw, ImageFont


def text_image(text, font_path, font_size, max_width):
        
        image = Image.new("RGBA", (max_width, font_size), (0,0,0,0))

        font = ImageFont.truetype(font_path, font_size)

        draw = ImageDraw.Draw(image)

        _, _, w, h = draw.textbbox((0,0), text, font=font)

        draw.text(((max_width - w)/2,0), text, font=font, fill="white", stroke_width=5, stroke_fill='black')

    
        # ImageDraw.Draw(image).text((0,0), text, font=font, fill="white")

        image = image.crop((0, 0, max_width, font_size,))

        image.save("text.png")



text_image("Hello World", "SweetieBubbleGum-Regular.ttf", 100, 1000)

exit()

def create_image(text, font_path, font_size, max_width):
            # Create a blank image with a white background
            image = Image.new("RGBA", (max_width, 1000), (0,0,0,0))
            draw = ImageDraw.Draw(image)

            # Load the font
            font = ImageFont.truetype(font_path, font_size)

            # Split the text into words
            words = text.split()

            # Initialize variables
            current_line = text#""
            y_position = 10

            text_max_width = max_width - 50

            # for word in words:
            #     # Check if adding the next word exceeds the max width
            #     text_width = draw.textlength(current_line + " " + word, font)
            #     text_height = font_size
            #     if text_width <= text_max_width:
            #         # If not, add the word to the current line
            #         if current_line:
            #             current_line += " "
            #         current_line += word
            #     else:
            #         # Calculate the X-position to center the text
            #         x_position = (text_max_width - draw.textlength(current_line, font)) // 2
            #         # Draw the current line at the calculated position
            #         draw.text((x_position, y_position), current_line, font=font, fill="white", stroke_width=5, stroke_fill='black')
            #         y_position += text_height
            #         current_line = word

            # Calculate the X-position for the last line to center the text
            x_position = (max_width - draw.textlength(current_line, font)) // 2
            # Draw the last line at the calculated position
            draw.text((x_position, y_position), current_line, font=font, fill="white", stroke_width=5, stroke_fill='black')

            # Crop the image to the actual content size
            image = image.crop((0, 0, max_width, y_position + font_size + 100))
            # return image
            
            return image

create_image(
      "Hello World!",
      "SweetieBubbleGum-Regular.ttf",
      100,
      1000
)

