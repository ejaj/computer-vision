import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.transform import hough_line_peaks


# DrawLine function from week 3
def DrawLine(l, shape):
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2] / q[2]
        if all(q >= 0) and all(q + 1 <= shape[1::-1]):
            return q

    lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1 - shape[1]], [0, 1, 1 - shape[0]]]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    if len(P) == 0:
        print("Line is completely outside image")
    else:
        plt.plot(*np.array(P).T)


# Load the image
image = cv2.imread('data/Box3.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Compute the Hough transform
hspace, angles, dists = transform.hough_line(edges)

# Find peaks in the Hough space
n = 10  # Number of peaks to find
extH, extAngles, extDists = hough_line_peaks(hspace, angles, dists, num_peaks=n)

# Plot the original edge-detected image
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Define the extent to get the correct units on the axes
extent = [np.rad2deg(angles[-1]), np.rad2deg(angles[0]), dists[-1], dists[0]]

# Plot the Hough space
plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + hspace), extent=extent, aspect='auto', cmap='hot')
plt.title('Hough Transform Accumulator')
plt.xlabel('Angles (degrees)')
plt.ylabel('Distance (pixels)')
plt.colorbar(label='Accumulator counts')

# Overlay the peaks on the Hough space
plt.scatter(np.rad2deg(extAngles), extDists, color='blue')
plt.title('Hough Transform with Peaks')

# Plot the identified lines on the original image
plt.subplot(1, 3, 3)
plt.imshow(image)
shape = image.shape
for angle, dist in zip(extAngles, extDists):
    rho = dist
    theta = angle
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    line = np.array([a, b, -rho])
    DrawLine(line, shape)
plt.title('Detected Lines on Image')

plt.tight_layout()
plt.show()
