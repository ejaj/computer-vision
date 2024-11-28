import numpy as np
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Define second-order derivatives
Gxx = np.array([[1, -2, 1], [0, 0, 0], [-1, 2, -1]], dtype=np.float32)  # Second derivative X
Gyy = np.array([[1, 0, -1], [-2, 0, 2], [1, 0, -1]], dtype=np.float32)  # Second derivative Y
Gxy = np.array([[1, -1, 0], [-1, 0, 1], [0, 1, -1]], dtype=np.float32)  # Mixed derivative

# Compute second-order gradients
gradient_xx = convolve(image, Gxx)
gradient_yy = convolve(image, Gyy)
gradient_xy = convolve(image, Gxy)

# Combine to steer the filter in a direction
theta = 45
theta_rad = np.radians(theta)
u = np.cos(theta_rad)
v = np.sin(theta_rad)

# Second-order steerable filter
steerable_gradient_2nd = (u ** 2) * gradient_xx + (2 * u * v) * gradient_xy + (v ** 2) * gradient_yy


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Orginal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.abs(steerable_gradient_2nd), cmap='gray')
plt.title(f"Second-Order Directional Gradient (θ = {theta}°)")
plt.axis("off")

plt.tight_layout()
plt.show()