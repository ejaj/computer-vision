import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'data/TestIm1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found or path is incorrect")

# Parameters for Harris detection
block_size = 2
ksize = 3
k = 0.04

# Find Harris corners
dst = cv2.cornerHarris(image, block_size, ksize, k)
dst = cv2.dilate(dst, None)  # Result is dilated for marking the corners

# Threshold for an optimal value, it may vary depending on the image.
image[dst > 0.01 * dst.max()] = 255

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.title('Harris Corners')
plt.axis('off')
plt.show()
