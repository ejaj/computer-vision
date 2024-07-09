import numpy as np
from scipy.spatial.transform._rotation import Rotation

# Define the intrinsic matrix K.
K = np.array([[1000, 0, 300],
              [0, 1000, 200],
              [0, 0, 1]])


def CrossOp(p):
    """
    Takes a vector p in 3D and returns the corresponding 3x3 skew-symmetric matrix
    for the cross product operation.
    """
    return np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])


# Define the rotation for Cam2 using the given angles.
angles = [0.7, -0.5, 0.8]  # angles in radians for rotation 'xyz'
R2 = Rotation.from_euler('xyz', angles).as_matrix()

# Define the translation for Cam2.
t2 = [0.2, 0, 1]

# Define the skew-symmetric matrix for the translation vector t2 using the CrossOp function
t2_skew = CrossOp(t2)

# Compute the essential matrix E using the formula: E = [t2]x R2
E = t2_skew @ R2

# The fundamental matrix F can be computed from the essential matrix E using the formula: F = K^-T * E * K^-1
# where K^-T is the inverse transpose of the intrinsic matrix K.

# Calculate the inverse of the intrinsic matrix K
K_inv = np.linalg.inv(K)

# Calculate the fundamental matrix F
F = K_inv.T @ E @ K_inv

print(F)
