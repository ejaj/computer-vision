import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Highlight edges and details	
Detects sharp transitions in the image.
What Happens:

Subtracting the blurred (low-pass) image from the original leaves high frequencies (edges).
Result: Edge-detected image.
"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Compute the Fourier Transform and shift the zero frequency component to the center
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# Create a low-pass filter mask
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2  # Center point
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # Pass low frequencies only
# Apply the mask
f_shift_low_pass = f_shift * mask
# Perform the inverse Fourier Transform
f_ishift = np.fft.ifftshift(f_shift_low_pass)
image_low_pass = np.fft.ifft2(f_ishift)
image_low_pass = np.abs(image_low_pass)

edges = image - image_low_pass

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Low-Pass Filtered Image')
plt.imshow(image_low_pass, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Edges (High-Pass Filtering)')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()