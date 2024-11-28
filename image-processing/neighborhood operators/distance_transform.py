import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Use Case: Computing distances from each pixel to the nearest background pixel (or nearest specific feature).	
Purpose: Measure spatial relationships or alignment.
"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Perform distance transform
distance_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)

# Normalize for visualization
distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display distance transform
plt.imshow(distance_transform_normalized, cmap='hot')
plt.title("Distance Transform")
plt.axis('off')
plt.show()