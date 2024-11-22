"""
Color transforms adjust brightness or balance across RGB channels. Manipulations can affect hue and saturation.
gR(x)=fR(x)+b
gG(x)=fG(x)+b
gB(x)=fB(x)+b
"""

import cv2
import matplotlib.pyplot as plt

color_image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/lena.png")

# Add brightness to each channel
brightened_image = cv2.convertScaleAbs(color_image, alpha=1, beta=50)

# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Brightened Image")
plt.imshow(cv2.cvtColor(brightened_image, cv2.COLOR_BGR2RGB))
plt.show()