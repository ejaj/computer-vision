import numpy as np
import matplotlib.pyplot as plt
from data_points import test_points
from sklearn.linear_model import RANSACRegressor

def estimate_line_homogeneous(p1, p2):
    """
    Estimates a line in homogeneous coordinates given two points.

    Args:
    p1 (tuple): The first point (x1, y1) in Cartesian coordinates.
    p2 (tuple): The second point (x2, y2) in Cartesian coordinates.

    Returns:
    numpy.ndarray: The coefficients of the line (a, b, c) in the form ax + by + c = 0.
    """

    p1_hom = np.array([p1[0], p1[1], 1])
    p2_hom = np.array([p2[0], p2[1], 1])

    # Calculate the cross product of the points
    line_coeffs = np.cross(p1_hom, p2_hom)

    return line_coeffs


n_inliers = 100
n_outliers = 50
points = test_points(n_inliers, n_outliers)

ransac = RANSACRegressor(random_state=0)
ransac.fit(points[0, :].reshape(-1, 1), points[1, :])

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(points[0, :].min(), points[0, :].max())
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

plt.figure(figsize=(10, 6))
plt.scatter(points[0, inlier_mask], points[1, inlier_mask], color='yellow', label='Inliers')
plt.scatter(points[0, outlier_mask], points[1, outlier_mask], color='red', label='Outliers')
plt.plot(line_X, line_y_ransac, color='green', linewidth=2, label='RANSAC regressor')
plt.title('RANSAC Line Fitting')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

point1 = (3, 4)
point2 = (6, 8)
line = estimate_line_homogeneous(point1, point2)
print("The line coefficients are:", line)