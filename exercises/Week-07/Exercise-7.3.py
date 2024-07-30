import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

# Load the image
image = cv2.imread('data/Box3.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Compute the Hough transform
hspace, angles, dists = transform.hough_line(edges)

# Plot the edge-detected image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Define the extent to get the correct units on the axes
extent = [np.rad2deg(angles[-1]), np.rad2deg(angles[0]), dists[-1], dists[0]]

# Plot the Hough space
plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + hspace), extent=extent, aspect='auto', cmap='hot')
plt.title('Hough Transform Accumulator')
plt.xlabel('Angles (degrees)')
plt.ylabel('Distance (pixels)')
plt.colorbar(label='Accumulator counts')

plt.tight_layout()
plt.show()
