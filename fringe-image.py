import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fringe image
image = cv2.imread("data/fringe.png", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Use edge detection to enhance the fringe lines
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Display the preprocessed image
plt.imshow(edges, cmap='gray')
plt.title("Edges of Fringe Image")
plt.show()


# Detect lines using Hough Line Transform
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)

# Draw the detected lines on the original image
line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the lines on the image
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines")
plt.show()


enhanced = cv2.equalizeHist(image)
plt.imshow(enhanced, cmap='gray')
plt.title("Enhanced Contrast")
plt.show()

# Apply adaptive thresholding
thresholded = cv2.adaptiveThreshold(
    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
plt.imshow(thresholded, cmap='gray')
plt.title("Adaptive Threshold")
plt.show()

# Detect lines on the thresholded image
edges = cv2.Canny(thresholded, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.title("Improved Line Detection")
plt.show()
