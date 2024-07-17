import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/TestIm1.png', cv2.IMREAD_GRAYSCALE)

# Calculate the x and y gradients using the Sobel operator
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y

# Calculate the magnitude
gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

# Calculate the direction
gradient_direction = np.arctan2(sobely, sobelx)

sobelx_abs = cv2.convertScaleAbs(sobelx)
sobely_abs = cv2.convertScaleAbs(sobely)
magnitude = cv2.convertScaleAbs(gradient_magnitude)

# Plotting
plt.figure(figsize=(15, 12))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(sobelx_abs, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(sobely_abs, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(gradient_direction, cmap='hsv')  # Using HSV to better visualize angles
plt.title('Gradient Direction')
plt.axis('off')

plt.tight_layout()
plt.show()
