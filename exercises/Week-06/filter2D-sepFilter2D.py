import cv2
import numpy as np

# Load an image
image = cv2.imread('data/house_input.png')

# Gaussian kernel for demonstration (3x3)
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1 / 16)

# Apply filter using filter2D
blurred = cv2.filter2D(image, -1, kernel)

# For sepFilter2D, separate the kernel into two 1D kernels (symmetrical Gaussian)
g = np.array([1, 2, 1]) * (1 / 4)
blurred_sep = cv2.sepFilter2D(image, -1, g, g)

# Display or save the images
cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred)
cv2.imshow('Blurred with Separable Filter', blurred_sep)
cv2.waitKey(0)
cv2.destroyAllWindows()
