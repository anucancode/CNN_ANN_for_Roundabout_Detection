import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from hog_layer import *
import time

start = time.time()

# Function to load and preprocess images
def load_images(image_paths, target_size=(512, 512)):
    images = []
    for path in image_paths:
        img = imread(path, as_gray=True)
        img_resized = resize(img, target_size)
        images.append(img_resized)
    return np.array(images)

# Load dataset
image_dir = "D:/BDSD MP Coding/CODE TRIAL 09 hog/data/sorted_unity_dataset/200"
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
labels = np.loadtxt("D:/BDSD MP Coding/CODE TRIAL 09 hog/data/sorted_unity_dataset/200/labels.txt")

# Load and preprocess images
images = load_images(image_paths)

# Reshape images for the model (adding the channel dimension)
images = images[..., np.newaxis]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the neural network model with HOG layer
model = Sequential([
    Input(shape=(512, 512, 1)),
    HOGLayer(pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3)  # Output layer for x, y, and radius
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mse'])

# Setup callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint('roundabout_detector_model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, callbacks=[checkpoint, early_stopping])

# Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}')

# Save the model
model.save('roundabout_detector_model_final.hdf5')

end = time.time()
print('Running time: %s Seconds' % (end - start))
