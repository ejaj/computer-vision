import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('data/Box3.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Create a blank image to draw the lines on
line_image = np.copy(image) * 0

# Draw lines on the image
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Merge the original image with the line image
combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

# Calculate the Hough accumulator space
accumulator = np.zeros((180, int(np.sqrt(gray.shape[0] ** 2 + gray.shape[1] ** 2))), dtype=np.uint64)
for y in range(edges.shape[0]):
    for x in range(edges.shape[1]):
        if edges[y, x] > 0:
            for theta in range(0, 180):
                rho = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))
                if 0 <= rho < accumulator.shape[1]:
                    accumulator[theta, rho] += 1

# Display the results
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image with Hough Lines')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')

plt.subplot(1, 3, 3)
plt.imshow(accumulator, cmap='hot', extent=[0, accumulator.shape[1], 180, 0], aspect='auto')
plt.title('Hough Transform Accumulator')
plt.xlabel('rho')
plt.ylabel('theta')

plt.show()
