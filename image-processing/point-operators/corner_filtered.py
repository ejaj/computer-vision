import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/603537.jpg", cv2.IMREAD_GRAYSCALE)

# Define a corner detection kernel
corner_kernel = np.array([[1, -2, 1],
                          [-2, 4, -2],
                          [1, -2, 1]], dtype=np.float32)

# Apply the corner detection filter
corner_filtered = cv2.filter2D(image, -1, corner_kernel)

plt.imshow(corner_filtered, cmap='gray')
plt.title("Corner Detection")
plt.axis("off")
plt.show()