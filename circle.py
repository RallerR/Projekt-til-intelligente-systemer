from PIL import Image, ImageDraw
import os
import random

# Parameters
num_images = 100  # Number of images to generate
image_size = 32   # Size of the image (32x32 pixels)
output_folder = 'CircleImages'

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(num_images):
    # Create a blank image with a white background
    image = Image.new('L', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(image)

    # Randomly choose radius and position for the circle
    radius = random.randint(5, image_size // 3)
    position = (random.randint(radius, image_size - radius), random.randint(radius, image_size - radius))

    # Draw a black circle periphery
    draw.ellipse([tuple(map(lambda x: x - radius, position)),
                  tuple(map(lambda x: x + radius, position))], outline='black')

    # Save the image
    image.save(f'{output_folder}/circle_{i}.jpg')

print(f'{num_images} circle periphery images generated in {output_folder}/')