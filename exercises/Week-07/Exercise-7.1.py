import cv2
from matplotlib import pyplot as plt
# Load the image
image = cv2.imread('data/Box3.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)


# Plot the edges
plt.figure(figsize=(10, 6))
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')
plt.show()