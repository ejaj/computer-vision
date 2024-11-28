import cv2
import numpy as np

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ehDpw.png", cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter for edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Display results
cv2.imshow("Sobel Filtered Image", sobel_combined / sobel_combined.max())
cv2.waitKey(0)
cv2.destroyAllWindows()
