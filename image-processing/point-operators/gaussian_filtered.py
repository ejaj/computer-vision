import cv2
import numpy as np

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/nois-lean.png", cv2.IMREAD_GRAYSCALE)

# Apply a Gaussian filter
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), sigmaX=2.0)

# Display results
cv2.imshow("Gaussian Filtered Image", gaussian_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
