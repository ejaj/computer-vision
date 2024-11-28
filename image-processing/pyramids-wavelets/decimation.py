import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/kodim08_grayscale.png", cv2.IMREAD_GRAYSCALE)

# Define filters (normalized to sum = 1)
filters = {
    'Linear [1, 2, 1]': np.array([1, 2, 1]) / 4,
    'Binomial [1, 4, 6, 4, 1]': np.array([1, 4, 6, 4, 1]) / 16,
    'Cubic (a=-1)': np.array([-1, 9, 16, 9, -1]) / 32,
    'Windowed Sinc': np.array([0.3, 0.4, 0.3]) 
}

# Function to apply a 1D filter in both directions (2D convolution)
def apply_filter(image, filter_kernel):
    # Convolve horizontally and then vertically
    filtered = convolve2d(image, filter_kernel[:, None], mode='same', boundary='symm')  # Horizontal
    filtered = convolve2d(filtered, filter_kernel[None, :], mode='same', boundary='symm')  # Vertical
    return filtered

# Downsample function
def downsample(image, filter_kernel, factor=2):
    filtered = apply_filter(image, filter_kernel)  # Apply low-pass filter
    downsampled = filtered[::factor, ::factor]    # Subsample every nth pixel
    return downsampled

# Perform decimation using different filters
downsampled_images = {}
for filter_name, filter_kernel in filters.items():
    downsampled_images[filter_name] = downsample(image, filter_kernel)

# Plot original and downsampled images
plt.figure(figsize=(12, 10))
plt.subplot(3, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

for i, (filter_name, downsampled_image) in enumerate(downsampled_images.items(), start=2):
    plt.subplot(3, 2, i)
    plt.title(filter_name)
    plt.imshow(downsampled_image, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
