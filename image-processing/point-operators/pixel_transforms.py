"""
Formula:

g(x)=af(x)+b

f(x): Input pixel value.
g(x): Output pixel value.
a: Gain (contrast adjustment).
b: Bias (brightness adjustment).
"""
import cv2
import matplotlib.pyplot as plt

input_image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/lena.png", cv2.IMREAD_GRAYSCALE)

gain = 1.2  # Contrast
bias = 50   # Brightness

adjusted_image = cv2.convertScaleAbs(input_image, alpha=gain, beta=bias)

# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(input_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Adjusted Image")
plt.imshow(adjusted_image, cmap='gray')
plt.show()

