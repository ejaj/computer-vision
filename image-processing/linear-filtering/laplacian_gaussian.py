import cv2
from scipy.ndimage import gaussian_laplace, convolve
import matplotlib.pyplot as plt
image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)
log_filtered = gaussian_laplace(image, sigma=1)

# Create a figure with two columns
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

# Display the processed image
axes[1].imshow(log_filtered, cmap='gray')
axes[1].set_title("Gaussian Laplace Image")
axes[1].axis("off")

# Adjust layout
plt.tight_layout()
plt.show()

