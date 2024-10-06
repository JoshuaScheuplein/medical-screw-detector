import os
from PIL import Image

# Define the directory containing subdirectories with images
base_directory = 'C:/Users/wagne/Desktop/imgs_used/resnet_alpha'

# Define the crop box (left, upper, right, lower)
crop_boxes = [(150, 150, 950, 950), (350, 200, 1150, 1000), (500, 300, 1150, 750) ] # Example crop box, adjust as needed

# Iterate over each directory in the base directory
for i, subdir in enumerate(os.listdir(base_directory)):
    subdir_path = os.path.join(base_directory, subdir)

    crop_box = crop_boxes[i]

    # Check if the path is a directory
    if os.path.isdir(subdir_path):
        # Iterate over each file in the subdirectory
        for file in os.listdir(subdir_path):

            print(file)

            if file.endswith('.png') and not file.__contains__('cropped'):
                file_path = os.path.join(subdir_path, file)

                # Open the image
                with Image.open(file_path) as img:
                    # Crop the image
                    cropped_img = img.crop(crop_box)

                    # Save the cropped image
                    output_file_path = os.path.join(subdir_path, f"{file[:-4]}_cropped_2.png")
                    cropped_img.save(output_file_path)