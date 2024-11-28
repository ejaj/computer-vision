import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Closing
When to Use:
To fill small holes or gaps in objects.
To connect small breaks or gaps in object boundaries.
Typical Applications:
Filling gaps: Improving object segmentation results.
Post-processing: Refining binary masks after segmentation.
"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Structuring element (3x3 box)
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, structuring_element)


# Display the gradients
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Orginal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(closed, cmap='gray')
plt.title("Closed Image")
plt.axis("off")

plt.tight_layout()
plt.show()
