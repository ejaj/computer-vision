import cv2
import numpy as np

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ehDpw.png", cv2.IMREAD_GRAYSCALE)

# Define a bilinear kernel
bilinear_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32)
bilinear_kernel /= bilinear_kernel.sum()

# Apply the bilinear filter
bilinear_filtered = cv2.filter2D(image, -1, bilinear_kernel)

# Display results
cv2.imshow("Bilinear Filtered Image", bilinear_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
