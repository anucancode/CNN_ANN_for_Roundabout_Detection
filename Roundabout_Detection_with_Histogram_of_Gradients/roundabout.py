import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time

start = time.time()

# Function to extract HOG features
def extract_hog_features(image_paths, target_size=(512, 512)):
    hog_features = []
    for path in image_paths:
        img = imread(path, as_gray=True)  # Load image in grayscale
        img_resized = resize(img, target_size)
        features, _ = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
        hog_features.append(features)
    return np.array(hog_features)

# Load dataset
image_dir = "D:/BDSD MP Coding/CODE TRIAL 09 hog/data/sorted_unity_dataset/1341ID preprocessed"  # Replace with actual directory path
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
labels = np.loadtxt("D:/BDSD MP Coding/CODE TRIAL 09 hog/data/sorted_unity_dataset/labels/RoundaboutInfo_1341ID_fixedY.txt")  # Assuming txt file with columns [x, y, radius]

# Extract HOG features
hog_features = extract_hog_features(image_paths, target_size=(512, 512))

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(hog_features.shape[1],)),
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