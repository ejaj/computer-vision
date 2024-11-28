from scipy.signal import resample

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/kodim08_grayscale.png", cv2.IMREAD_GRAYSCALE)

scale = 4

# Resample image using windowed sinc (1D across rows and columns)
resampled_image = resample(image, image.shape[0] * scale, axis=0)  # Vertical interpolation
resampled_image = resample(resampled_image, image.shape[1] * scale, axis=1)  # Horizontal

# Display windowed sinc results
plt.figure(figsize=(6, 6))
plt.title("Windowed Sinc Interpolation")
plt.imshow(resampled_image, cmap="gray")
plt.axis("off")
plt.show()
