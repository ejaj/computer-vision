import cv2
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/lena.png", cv2.IMREAD_GRAYSCALE)


# Gamma correction
gamma = 2.0
gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Gamma Corrected")
plt.imshow(gamma_corrected, cmap='gray')
plt.show()
