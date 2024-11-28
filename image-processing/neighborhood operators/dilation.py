import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Dilation
When to Use:
To expand regions of interest (foreground pixels, value 1).
To fill small holes or gaps along the boundaries of objects.
Typical Applications:
Object detection: Making small objects more prominent.
Edge linking: Connecting broken edges in images.

"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Structuring element (3x3 box)
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated_img = cv2.dilate(image, structuring_element)

# Display the gradients
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Orginal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dilated_img, cmap='gray')
plt.title("Dilated Image")
plt.axis("off")

plt.tight_layout()
plt.show()
