import os
import cv2
import numpy as np
from PIL import Image


# Load images from a directory using OpenCV
def load_images_from_folder_cv2(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


# Resize image for testing
def resize_image(image, scale=0.25):
    return cv2.resize(image, None, fx=scale, fy=scale)


# Detect checkerboard corners
def detect_checkerboard(image, pattern_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    return ret, corners


# Load images
folder_path = 'data/chess'
images, filenames = load_images_from_folder_cv2(folder_path)

# Test on a resized image to find the correct pattern size
pattern_size = (5, 5)

# Resize the first image to a smaller resolution for testing
im_small = resize_image(images[0])

# Detect checkerboard on the small image
ret, corners = detect_checkerboard(im_small, pattern_size)

if ret:
    print("Checkerboard detected in the resized test image.")
else:
    print("Checkerboard not detected in the resized test image. Adjust the pattern size and try again.")

# Detect checkerboards in all images
successful_images = []
successful_filenames = []
for img, fname in zip(images, filenames):
    ret, corners = detect_checkerboard(img, pattern_size)
    if ret:
        successful_images.append(img)
        successful_filenames.append(fname)
        print(f"Checkerboard detected in image: {fname}")
    else:
        print(f"Checkerboard not detected in image: {fname}")

print(f"Number of images with detected checkerboards: {len(successful_images)}")
