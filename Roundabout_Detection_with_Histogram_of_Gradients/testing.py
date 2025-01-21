import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.feature import hog
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Function to extract HOG features
def extract_hog_features(image_paths, target_size=(128, 128)):
    hog_features = []
    for path in image_paths:
        img = imread(path, as_gray=True)  # Load image in grayscale
        img_resized = resize(img, target_size)
        features, _ = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
        hog_features.append(features)
    return np.array(hog_features)

# Load dataset
image_dir = "D:/BDSD MP Coding/CODE TRIAL 09 hog/data/20"   # Replace with actual directory path
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
#labels = np.loadtxt('path_to_labels.txt')  # Assuming txt file with columns [x, y, radius]

# Extract HOG features
hog_features = extract_hog_features(image_paths, target_size=(512, 512))

# Load the trained model
model = load_model('roundabout_detector_model_final.hdf5')

# Make predictions
predictions = model.predict(hog_features)

# Create output directory for images with drawn circles and labels
output_dir = 'output_images_with_circles_and_labels'
os.makedirs(output_dir, exist_ok=True)

# Visualize predictions and save images with drawn circles and labels
for i, path in enumerate(image_paths):
    img = imread(path)
    
    # Draw predicted circle
    x, y, radius = predictions[i]
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    circle = Circle((x, y), radius, color='r', fill=False)
    ax.add_patch(circle)
    ax.axis('off')
    
    # Save image with drawn circle
    output_path = os.path.join(output_dir, os.path.basename(path))
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save predicted labels as text file
    labels_output_path = os.path.join(output_dir, f'{os.path.basename(path)}.txt')
    with open(labels_output_path, 'w') as f:
        f.write(f'Predicted Labels:\nX: {x}\nY: {y}\nRadius: {radius}')
