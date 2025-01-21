import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from natsort import *

# Path to your data file and image folder
data_file_path = "D:/BDSD MP Coding/new dataset unity/RoundaboutInfo_combined_fixedY.txt"
image_folder_path = "D:/BDSD MP Coding/new dataset unity/train_new/train_new"
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

# Get sorted image files
image_files = natsorted([img for img in os.listdir(image_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])

for image_file, (x, y, r) in zip(image_files, circles):
    image_path = os.path.join(image_folder_path, image_file)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw the circle
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline='white', width=2)

    # Save the modified image
    output_image_path = os.path.join(output_folder_path, image_file)
    image.save(output_image_path)

print("Processing complete. Modified images are saved in:", output_folder_path)

# Display the first processed image for verification
plt.imshow(image)
plt.show()
