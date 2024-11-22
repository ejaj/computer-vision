"""
Compositing combines two images using an alpha matte to define transparency.

Math Explanation:

C(x)=(1−α)B(x)+αF(x)
C(x): Composite pixel value.
B(x): Background pixel value.
F(x): Foreground pixel value.
α: Transparency value (0 to 1).
"""

import cv2
import matplotlib.pyplot as plt

background = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/boat.tiff")
foreground = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/lena.png")

# Alpha blending
alpha = 0.5  # Transparency
composite_image = cv2.addWeighted(foreground, alpha, background, 1 - alpha, 0)

# Display images
plt.imshow(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))
plt.title("Composite Image")
plt.show()