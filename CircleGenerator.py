from PIL import Image, ImageDraw
import os
import random

def generate_random_circle_images(output_folder, num_images):
    image_size = 32  # Image size (32x32 pixels)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_images):
        # Create a blank white image
        img = Image.new('1', (image_size, image_size), color=1)  # '1' for 1-bit pixels, black and white

        # Create a drawing context
        draw = ImageDraw.Draw(img)

        # Define the maximum radius of the circle so it stays within the image
        max_radius = image_size // 4  # Adjust this value as needed

        # Randomly choose a radius
        radius = random.randint(5, image_size // 3)

        # Adjusted circle parameters to ensure the circle is completely within the image
        margin = radius + 1  # Ensure there's space for the circle
        center_x = random.randint(margin, image_size - margin)
        center_y = random.randint(margin, image_size - margin)

        # Draw the circle's edge in black
        draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=0)

        # Generate a unique filename for each image
        filename = os.path.join(output_folder, f'circle_{i}.png')

        # Save the binary image
        img.save(filename)

# Example usage
output_folder = 'Dataset1'  # Specify the folder where images will be saved
num_images = 1000  # Specify the number of random circle images to generate

generate_random_circle_images(output_folder, num_images)