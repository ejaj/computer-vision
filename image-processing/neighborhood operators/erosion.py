import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Erosion
When to Use:
To shrink regions of interest.
To remove small noise or detach objects that are connected by thin "bridges."
Typical Applications:
Noise removal: Eliminating small, irrelevant details.
Edge refinement: Reducing the thickness of detected boundaries.
"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Structuring element (3x3 box)
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
eroded_img = cv2.erode(image, structuring_element)

# Display the gradients
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Orginal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(eroded_img, cmap='gray')
plt.title("Erode Image")
plt.axis("off")

plt.tight_layout()
plt.show()
