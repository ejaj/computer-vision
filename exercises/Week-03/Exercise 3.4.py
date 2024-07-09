# Define the intrinsic matrix K.
import numpy as np
from scipy.spatial.transform._rotation import Rotation

K = np.array([[1000, 0, 300],
              [0, 1000, 200],
              [0, 0, 1]])

# Define the 3D point Q.
Q = np.array([[1],
              [0.5],
              [4],
              [1]])


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

# For Cam1, since the rotation is identity and translation is zero, the projection is straightforward
# Define the extrinsic parameters for Cam1 (identity rotation and zero translation)
R1 = np.eye(3)
t1 = np.array([0, 0, 0])

# Define the projection matrix for Cam1
P1 = K @ np.hstack((R1, t1.reshape(-1, 1)))

# Project the point Q onto the image plane of Cam1
q1 = P1 @ Q

# Normalize the projected point q1
q1 = q1 / q1[2]

# For Cam2, we use the rotation R2 and translation t2 calculated earlier
# Convert translation vector t2 into a column vector and concatenate with R2 to form the extrinsic matrix for Cam2
t2_col = np.array(t2).reshape(-1, 1)
extrinsic_matrix_cam2 = np.hstack((R2, t2_col))

# Define the projection matrix for Cam2
P2 = K @ extrinsic_matrix_cam2

# Project the point Q onto the image plane of Cam2
q2 = P2 @ Q

# Normalize the projected point q2
q2 = q2 / q2[2]

# Correcting the computation for the homogeneous coordinates of q1
q1_homogeneous_corrected = np.array([[q1[0][0]], [q1[1][0]], [1]])  # Ensure q1 is in homogeneous coordinates correctly

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

# Recompute the epipolar line l for q1 in camera two using the corrected homogeneous coordinates
l_corrected = F @ q1_homogeneous_corrected

print(l_corrected)
