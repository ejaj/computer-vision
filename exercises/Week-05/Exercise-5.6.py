import os
from PIL import Image
import numpy as np


# Load images from a directory
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


# Check if all images have the same dimensions
def check_image_dimensions(images):
    dimensions = [img.size for img in images]
    unique_dimensions = set(dimensions)
    if len(unique_dimensions) > 1:
        return False, unique_dimensions
    return True, unique_dimensions


# Load images
folder_path = 'data/chess'  # Adjust the path to your folder
images, filenames = load_images_from_folder(folder_path)

# Check dimensions
all_same_size, dimensions = check_image_dimensions(images)
print(f"All images have the same dimensions: {all_same_size}")
if not all_same_size:
    print(f"Unique dimensions found: {dimensions}")

# Discard images with different dimensions
if not all_same_size:
    target_size = images[0].size
    images = [img for img in images if img.size == target_size]
    filenames = [filenames[i] for i in range(len(filenames)) if images[i].size == target_size]

print(f"Number of images after discarding: {len(images)}")
