import cv2
import numpy as np
import matplotlib.pyplot as plt

# Simulated coordinates of ×-marks in two images (image A and image B)
# Replace these coordinates with the ones you get from plt.ginput()
points_a = np.array([[100, 200], [200, 200], [200, 300], [100, 300]], dtype='float32')  # Image A
points_b = np.array([[150, 250], [250, 250], [250, 350], [150, 350]], dtype='float32')  # Image B

# Estimate the homography matrix from points in image A to points in image B
H, status = cv2.findHomography(points_a, points_b)

# Suppose we want to transform a new point from image A (e.g., the location of the small object)
point_a = np.array([150, 250, 1])  # New point in image A (x, y, 1 for homogeneous coordinates)

# Transform this point to image B's coordinate system using the homography matrix
point_b_transformed = np.dot(H, point_a)
point_b_transformed = point_b_transformed / point_b_transformed[2]  # Convert back from homogeneous coordinates

print(f"Point in image A: {point_a[:2]}")
print(f"Transformed point in image B: {point_b_transformed[:2]}")

# Optional: Visualize the points in a plot (for illustration)
# Visualizing original points in image A and their corresponding points in image B
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(points_a[:, 0], points_a[:, 1], color='red', label='Original Points A')
plt.scatter(point_a[0], point_a[1], color='blue', label='New Point A')
plt.title('Image A Points')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(points_b[:, 0], points_b[:, 1], color='green', label='Corresponding Points B')
plt.scatter(point_b_transformed[0], point_b_transformed[1], color='orange', label='Transformed New Point B')
plt.title('Image B Points')
plt.legend()

plt.show()
