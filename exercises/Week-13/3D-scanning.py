import numpy as np
import cv2

c = np.load('data/casper/calib.npy', allow_pickle=True).item()
image = cv2.imread("data/casper/sequence/frames0_0.png", cv2.IMREAD_GRAYSCALE)

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
