import numpy as np

"""
K = [
[fx, s, cx],
[0, fy, cy],
[0, 0, 1]
]
resolution: 1920 × 1080
f = 1000 (focal length in mm)
cx = 1920/2
cy = 1080/2 
s = 0 skew coefficien
"""

# Define the intrinsic matrix K
K = np.array([
    [1000, 0, 960],
    [0, 1000, 540],
    [0, 0, 1]
])

# Define the rotation matrix R
R = np.array([
    [np.sqrt(1 / 2), -np.sqrt(1 / 2), 0],
    [np.sqrt(1 / 2), np.sqrt(1 / 2), 0],
    [0, 0, 1]
])
# Define the translation vector t
t = np.array([[0], [0], [10]])
# Combine R and t into a 3x4 matrix
Rt = np.hstack((R, t))

# Compute the projection matrix P
P = K @ Rt  # Using the @ operator for matrix multiplication

# Define some 3D points (in homogeneous coordinates)
points_3D = np.array([
    [0, 0, 0, 1],  # Q000
    [1, 0, 0, 1],  # Q100
    [0, 1, 0, 1],  # Q010
    [0, 0, 1, 1],  # Q001
    [1, 1, 0, 1],  # Q110
    [1, 0, 1, 1],  # Q101
    [0, 1, 1, 1],  # Q011
    [1, 1, 1, 1]  # Q111
])

# Project the points using the projection matrix P
projected_points = P @ points_3D.T  # Transpose points to match dimensions for multiplication

# Normalize the projected points to convert from homogeneous coordinates to 2D
projected_points_normalized = projected_points[:2] / projected_points[2]

# Print the results
print("Projected 2D points:")
print(projected_points_normalized.T)  # Transpose back for easier reading
