import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform

img = cv2.imread('data/Box3.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Compute the Hough transform
hspace, angles, dists = transform.hough_line(edges)
# Plot the original edge-detected image
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection using Canny')
plt.axis('off')

# Plot the Hough space
plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + hspace), extent=[np.rad2deg(angles[-1]), np.rad2deg(angles[0]), dists[-1], dists[0]], cmap='hot',
           aspect='auto')
plt.title('Hough Transform Accumulator')
plt.xlabel('Angles (degrees)')
plt.ylabel('Distance (pixels)')
plt.colorbar(label='Accumulator counts')

plt.tight_layout()
plt.show()
