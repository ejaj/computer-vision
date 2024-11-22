import cv2
import numpy as np

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ehDpw.png", cv2.IMREAD_GRAYSCALE)

kernel_size = 5
box_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

# print(box_kernel)
# Apply the box filter
box_filtered = cv2.filter2D(image, -1, box_kernel)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Box Filtered Image", box_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()