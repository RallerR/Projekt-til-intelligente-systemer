from PIL import Image
import os

def process_images(input_folder, output_folder, base_name="image"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counter = 0  # Initialize a counter for naming the files
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to greyscale
            img = img.resize((64, 64), Image.ANTIALIAS)

            # Construct the new filename
            new_filename = f"{base_name}.{counter}.jpg"
            output_path = os.path.join(output_folder, new_filename)
            img.save(output_path)

            counter += 1  # Increment the counter for the next file

# Specify your directories here
input_folder = 'test'  # Change to your directory with images
output_folder = 'test'  # Change to your desired output directory

process_images(input_folder, output_folder)