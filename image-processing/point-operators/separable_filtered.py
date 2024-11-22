import numpy as np
import cv2
from scipy.ndimage import convolve1d

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/lena.png", cv2.IMREAD_GRAYSCALE)

# Gaussian kernel (1D)
def gaussian_kernel_1d(size, sigma):
    """Generates a 1D Gaussian kernel."""
    kernel = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.exp(-0.5 * (kernel / sigma)**2)
    return kernel / kernel.sum()

# Apply separable Gaussian filter
def separable_filter(image, kernel):
    """Applies separable filtering using a 1D kernel."""
    # Horizontal convolution (row-wise)
    temp = convolve1d(image, kernel, axis=1, mode='reflect')
    # Vertical convolution (column-wise)
    result = convolve1d(temp, kernel, axis=0, mode='reflect')
    return result

# Define kernel size and standard deviation
kernel_size = 5
sigma = 1.0

# Generate 1D Gaussian kernel
kernel = gaussian_kernel_1d(kernel_size, sigma)

# Apply separable filtering
filtered_image = separable_filter(image, kernel)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Separable Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()