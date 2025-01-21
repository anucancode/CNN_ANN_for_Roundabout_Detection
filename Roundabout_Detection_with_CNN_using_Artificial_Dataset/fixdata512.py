def adjust_labels(input_file, output_file, image_height):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            x, y, radius = map(float, line.split())
            adjusted_y = image_height - y
            file.write(f"{x} {adjusted_y} {radius}\n")

# Usage
input_file = './data/resized_data_512x512.txt'
output_file = './data/fixedYdata512.txt'
image_height = 512
adjust_labels(input_file, output_file, image_height)
