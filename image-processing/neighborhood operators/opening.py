import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Opening
When to Use:
To remove small noise while preserving the overall shape and size of larger objects.
Typical Applications:
Preprocessing: Cleaning up noisy binary images before further analysis.
Object isolation: Removing small speckles around larger objects.
"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Structuring element (3x3 box)
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, structuring_element)


# Display the gradients
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Orginal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(opened, cmap='gray')
plt.title("Opend Image")
plt.axis("off")

plt.tight_layout()
plt.show()
