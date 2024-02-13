import numpy as np
import itertools as it


# Given box3d function
def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    return np.hstack(points) / 2


# Given project_points function
def project_points(K, R, t, Q):
    RT = np.hstack((R, t))
    P = K @ RT
    homogenous_coordinates = P @ Q
    return homogenous_coordinates[:-1] / homogenous_coordinates[-1]


# Generate the 3D points for the cube
cube_points = box3d()

# Convert points to homogeneous coordinates by adding a row of ones
homogeneous_cube_points = np.vstack((cube_points, np.ones(cube_points.shape[1])))

# Define the camera intrinsic matrix K
f = 600  # focal length
cx = cy = 400  # principal point offset
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])

# Define the rotation matrix R and the translation vector t
R = np.eye(3)
t = np.array([[0], [0.2], [1.5]])

# Project all the points onto the image plane
projected_points = project_points(K, R, t, homogeneous_cube_points)

# Check if all points are within the reasonable range of an image sensor
# We will consider a common resolution like 1920x1080 as a reference
# However, for the principal point offset given, we'll use twice the offsets
# as the assumed resolution width and height to check if the points are within the range
resolution_x = cx * 2
resolution_y = cy * 2

# Check if any of the points fall outside this assumed resolution
points_within_sensor = np.all((projected_points[0] >= 0) & (projected_points[0] <= resolution_x) &
                              (projected_points[1] >= 0) & (projected_points[1] <= resolution_y))

print(projected_points)
print(points_within_sensor)
