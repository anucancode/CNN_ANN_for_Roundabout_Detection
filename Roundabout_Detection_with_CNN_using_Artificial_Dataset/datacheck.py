import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Path to your data file and image folder
data_file_path = "D:/BDSD MP Coding/new dataset unity/labelfiles/combined_labels.txt"
image_folder_path = "D:/BDSD MP Coding/new dataset unity/0455ID preprocessed"
output_folder_path = './data/check'

# Ensure the output directory exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Read circle data
circles = []
with open(data_file_path, 'r') as file:
    for line in file:
        x, y, r = map(float, line.strip().split())
        circles.append((x, y, r))

# Process images
image_files = sorted([img for img in os.listdir(image_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])

for image_file, (x, y, r) in zip(image_files, circles):
    image_path = os.path.join(image_folder_path, image_file)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Adjust these scales based on your specific dataset and image dimensions
    scale_x = 1.0
    scale_y = 1.0
    scale_r = 1.0

    # Modify these if the origin is not the top-left corner
    adjusted_x = x * scale_x
    adjusted_y = y * scale_y  # Inverting Y if origin is bottom-left
    adjusted_r = r * scale_r

    # Draw the circle
    draw.ellipse([(adjusted_x - adjusted_r, adjusted_y - adjusted_r),
                  (adjusted_x + adjusted_r, adjusted_y + adjusted_r)], 
                  outline='white', width=2)

    # Save the modified image
    output_image_path = os.path.join(output_folder_path, image_file)
    image.save(output_image_path)

print("Processing complete. Modified images are saved in:", output_folder_path)