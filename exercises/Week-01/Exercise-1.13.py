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


def project_points(K, R, t, Q):
    # Create the 3x4 RT matrix by concatenating R and t
    RT = np.hstack((R, t))
    # Project the 3D points to 2D points using the camera matrix K and the RT matrix
    P = K @ RT
    # Multiply the projection matrix P with the 3D points Q to get homogeneous coordinates
    homogenous_coordinates = P @ Q
    # Convert to 2D inhomogeneous coordinates by dividing by the last row
    projected_points = homogenous_coordinates[:-1] / homogenous_coordinates[-1]
    return projected_points


# Define camera matrix K, rotation matrix R, and translation vector t
K = np.eye(3)  # Identity matrix for camera matrix K
R = np.eye(3)  # Identity matrix for camera pose R
t = np.array([[0], [0], [4]])  # Translation vector

# Generate 3D points using the box3d function
Q = box3d()

# Project the 3D points to 2D points
projected_points = project_points(K, R, t, Q)

# Plotting
fig = plt.figure(figsize=(12, 6))

# Plot 3D points
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(Q[0, :], Q[1, :], Q[2, :], c='b', marker='o')
ax1.set_title('3D Box Points')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot 2D projection
ax2 = fig.add_subplot(122)
ax2.scatter(projected_points[0, :], projected_points[1, :], c='r', marker='o')
ax2.set_title('2D Projected Points')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')

plt.show()
