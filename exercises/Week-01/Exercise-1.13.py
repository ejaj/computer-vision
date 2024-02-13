import numpy as np
import itertools as it


def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    return np.hstack(points) / 2


def project_points(K, R, t, Q):
    RT = np.hstack((R, t))
    P = K @ RT
    homogenous_coordinates = P @ Q
    return homogenous_coordinates[:-1] / homogenous_coordinates[-1]


K = np.eye(3)  # Identity matrix for camera matrix K
R = np.eye(3)  # Identity matrix for camera pose R
t = np.array([[0], [0], [4]])
Q = box3d()  # Generate 3D points using your box3d function

projected_points = project_points(K, R, t, Q)

print("Projected 2D Points:")
print(projected_points)
