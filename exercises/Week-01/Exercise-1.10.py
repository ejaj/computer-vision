import matplotlib.pyplot as plt
import cv2

im = cv2.imread("data/kazi.jpg")
im = im[:, :, ::-1]
plt.imshow(im)
plt.show()  # Display the image
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Close all OpenCV windows
