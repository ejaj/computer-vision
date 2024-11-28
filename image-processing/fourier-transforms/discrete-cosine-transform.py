import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Compress the image by retaining only low-frequency components.

What Happens:

The DCT transforms the image into frequency components.
Most information is retained in the low-frequency components (top-left of DCT image).
The image can be reconstructed using only significant coefficients for compression.


"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Apply Discrete Cosine Transform
image_float = np.float32(image) / 255.0  # Normalize the image
dct = cv2.dct(image_float)  # Perform DCT
log_dct = np.log(abs(dct) + 1)  # Log transform for visualization

# Reconstruct the image using Inverse DCT
idct = cv2.idct(dct)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('DCT Coefficients (Log Scale)')
plt.imshow(log_dct, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Reconstructed Image')
plt.imshow(idct, cmap='gray')
plt.axis('off')

plt.show()