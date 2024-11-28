import numpy as np
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # Sobel X
Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)   # Sobel Y


# Compute gradients
gradient_x = convolve(image, Gx)
gradient_y = convolve(image, Gy)


# Display the gradients
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Orginal")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(np.abs(gradient_x), cmap='gray')
plt.title("Gradient X (Horizontal)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(np.abs(gradient_y), cmap='gray')
plt.title("Gradient Y (Vertical)")
plt.axis("off")

plt.tight_layout()
plt.show()


def steerable_filter(gradient_x, gradient_y, theta):
    """
    Apply steerable filter to compute gradients in a specific direction.
    theta: Orientation in degrees
    """
    # Convert theta to radians
    theta_rad = np.radians(theta)
    
    # Unit direction vector
    u = np.cos(theta_rad)
    v = np.sin(theta_rad)
    
    # Compute directional gradient
    directional_gradient = u * gradient_x + v * gradient_y
    return directional_gradient

# Compute the directional gradient at 45 degrees
theta = 45
directional_gradient = steerable_filter(gradient_x, gradient_y, theta)

# Display the directional gradient
plt.imshow(np.abs(directional_gradient), cmap='gray')
plt.title(f"Directional Gradient (θ = {theta}°)")
plt.axis("off")
plt.show()