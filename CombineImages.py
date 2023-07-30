import os
import cv2
import numpy as np

IMAGE_DIR = './images'
OUTPUT_DIR = './output'
COMBINED_DIR = './Combined'

# Create combined directory if it doesn't exist
if not os.path.exists(COMBINED_DIR):
    os.makedirs(COMBINED_DIR)

# List all files in the directory
image_files = os.listdir(IMAGE_DIR)
output_files = os.listdir(OUTPUT_DIR)

# Find matching files in both directories
for image_file in image_files:
    if image_file in output_files:
        # Read original and output images
        img_original = cv2.imread(os.path.join(IMAGE_DIR, image_file))
        img_output = cv2.imread(os.path.join(OUTPUT_DIR, image_file))

        if img_original is None or img_output is None:
            print(f"Unable to read image: {image_file}")
            continue

        # Resize both images to have the same height
        height = max(img_original.shape[0], img_output.shape[0])
        img_original = cv2.resize(img_original, (img_original.shape[1], height))
        img_output = cv2.resize(img_output, (img_output.shape[1], height))

        # Concatenate images horizontally
        img_combined = np.concatenate((img_original, img_output), axis=1)

        # Write combined image to the Combined directory
        cv2.imwrite(os.path.join(COMBINED_DIR, image_file), img_combined)
