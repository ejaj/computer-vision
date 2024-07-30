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

# Find peaks in the Hough space
from skimage.transform import hough_line_peaks

# Define number of peaks to find
n = 10
extH, extAngles, extDists = hough_line_peaks(hspace, angles, dists, num_peaks=n)

# Plot the edge-detected image
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Define the extent to get the correct units on the axes
extent = [np.rad2deg(angles[-1]), np.rad2deg(angles[0]), dists[-1], dists[0]]

# Plot the Hough space
plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + hspace), extent=extent, aspect='auto', cmap='hot')
plt.title('Hough Transform Accumulator')
plt.xlabel('Angles (degrees)')
plt.ylabel('Distance (pixels)')
plt.colorbar(label='Accumulator counts')

# Overlay the peaks on the Hough space
plt.scatter(np.rad2deg(extAngles), extDists, color='blue')
plt.title('Hough Transform with Peaks')

# Plot the identified lines on the original image
plt.subplot(1, 3, 3)
plt.imshow(image)
for angle, dist in zip(extAngles, extDists):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    plt.plot((0, image.shape[1]), (y0, y1), '-r')
plt.xlim((0, image.shape[1]))
plt.ylim((image.shape[0], 0))
plt.title('Detected Lines on Image')

plt.tight_layout()
plt.show()
