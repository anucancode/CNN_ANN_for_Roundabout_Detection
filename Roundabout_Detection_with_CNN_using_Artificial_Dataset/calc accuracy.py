import numpy as np

def read_data(file_path):
    with open(file_path, "r") as file:
        data = np.array([list(map(float, line.strip().split())) for line in file])
    return data

def calculate_accuracy(original_data, cnn_data, coordinate_tolerance, radius_tolerance):
    # Calculate Euclidean distances for (x, y) coordinates
    distances = np.sqrt(np.sum((original_data[:, :2] - cnn_data[:, :2]) ** 2, axis=1))
    # Calculate differences in radii
    radii_differences = np.abs(original_data[:, 2] - cnn_data[:, 2])
    
    # Check if predictions are within the tolerances
    accurate_coordinates = distances <= coordinate_tolerance
    accurate_radii = radii_differences <= radius_tolerance
    
    # Overall accuracy (both coordinates and radii must be within tolerance)
    overall_accurate = np.logical_and(accurate_coordinates, accurate_radii)
    return np.mean(overall_accurate) * 100  # Convert to percentage

def main():
    # Paths to the data files
    original_data_file = "data/resized_data_512x512.txt"  # Update with the path to your original data file
    cnn_data_file = "D:/BDSD MP Coding/code trial 03/data/test/results/result.txt"         # Update with the path to your CNN results file

    # Read the data
    original_data = read_data(original_data_file)
    cnn_data = read_data(cnn_data_file)

    # Define tolerance levels (adjust these as needed)
    coordinate_tolerance = 5.0  # pixels
    radius_tolerance = 5.0      # pixels

    # Calculate and print the accuracy
    accuracy_percent = calculate_accuracy(original_data, cnn_data, coordinate_tolerance, radius_tolerance)
    print(f"Accuracy: {accuracy_percent:.2f}%")

if __name__ == "__main__":
    main()
