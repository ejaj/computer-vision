import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    points = np.hstack(points)
    # Convert points to homogeneous coordinates by adding a row of ones
    points_homogeneous = np.vstack((points, np.ones(points.shape[1])))
    return points_homogeneous / 2


def project_points(K, R, t, Q, distCoeffs=None):
    if distCoeffs is None:
        distCoeffs = []

    # Create the 3x4 RT matrix by concatenating R and t
    RT = np.hstack((R, t))
    # Project the 3D points to 2D points using the camera matrix K and the RT matrix
    P = K @ RT
    # Multiply the projection matrix P with the 3D points Q to get homogeneous coordinates
    homogenous_coordinates = P @ Q
    # Convert to 2D inhomogeneous coordinates by dividing by the last row
    projected_points = homogenous_coordinates[:-1] / homogenous_coordinates[-1]

    if distCoeffs:
        # Compute the radial distance from the center of the image
        x = projected_points[0, :]
        y = projected_points[1, :]
        r2 = x ** 2 + y ** 2

        # Apply the radial distortion
        radial_distortion = np.zeros_like(x)
        for i, k in enumerate(distCoeffs):
            radial_distortion += k * (r2 ** (2 * (i + 1)))

        # Apply the distortion to the projected points
        x_distorted = x * (1 + radial_distortion)
        y_distorted = y * (1 + radial_distortion)

        projected_points = np.vstack((x_distorted, y_distorted))

    return projected_points


# Given parameters for Exercise 2.1
f = 600
alpha = 1
beta = 0
delta_x = 400
delta_y = 400

# Define the camera matrix K
K = np.array([
    [f, alpha, delta_x],
    [0, f, delta_y],
    [0, 0, 1]
])

# Define the rotation matrix R (identity) and translation vector t
R = np.eye(3)
t = np.array([[0], [0.2], [1.5]])

# Generate 3D points using the box3d function
Q = box3d()

# Define the distortion coefficients
distCoeffs = [-0.2]

# Project the 3D points to 2D points with distortion
projected_points_distorted = project_points(K, R, t, Q, distCoeffs)

# Define image resolution
resolution_x = 800  # width in pixels
resolution_y = 800  # height in pixels

# Check if all points are within the image sensor bounds
within_bounds_distorted = np.all(
    (projected_points_distorted[0, :] >= 0) & (projected_points_distorted[0, :] < resolution_x) &
    (projected_points_distorted[1, :] >= 0) & (projected_points_distorted[1, :] < resolution_y))

print("Are all points captured by the image sensor (with distortion)?", within_bounds_distorted)

# Determine where the corner point (-0.5, -0.5, -0.5) projects to with distortion
corner_point = np.array([[-0.5], [-0.5], [-0.5], [1]])
projected_corner_distorted = project_points(K, R, t, corner_point, distCoeffs)

print("Projection of corner point (-0.5, -0.5, -0.5) with distortion:")
print(projected_corner_distorted)
