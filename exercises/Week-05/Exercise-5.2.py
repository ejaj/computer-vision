import numpy as np


def triangulate_n(qs, Ps):
    """
    Triangulate a single 3D point from its projections in n views using a linear algorithm.

    Parameters:
    - qs: List of n pixel coordinates in homogeneous form (x, y, 1).
    - Ps: List of n projection matrices.

    Returns:
    - The 3D coordinates of the triangulated point in homogeneous form.
    """
    A = np.zeros((len(qs) * 2, 4))

    for i, (q, P) in enumerate(zip(qs, Ps)):
        A[2 * i] = q[0] * P[2] - P[0]
        A[2 * i + 1] = q[1] * P[2] - P[1]

    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]

    return X / X[-1]


# Define the projection matrices
K = np.array([
    [700, 0, 600],
    [0, 700, 400],
    [0, 0, 1]
])

R = np.eye(3)
t1 = np.array([0, 0, 1]).reshape((3, 1))
t2 = np.array([0, 0, 20]).reshape((3, 1))

P1 = K @ np.hstack((R, t1))
P2 = K @ np.hstack((R, t2))

# Define the original 3D point Q
Q = np.array([1, 1, 0, 1]).reshape((4, 1))

# Compute the projections q1 and q2
q1_homogeneous = P1 @ Q
q2_homogeneous = P2 @ Q

q1 = q1_homogeneous[:2] / q1_homogeneous[2]
q2 = q2_homogeneous[:2] / q2_homogeneous[2]

# Add noise to the projections
q1_tilde = q1 + np.array([1, -1]).reshape((2, 1))
q2_tilde = q2 + np.array([1, -1]).reshape((2, 1))

# Convert to homogeneous coordinates
q1_tilde_homogeneous = np.vstack((q1_tilde, np.array([1])))
q2_tilde_homogeneous = np.vstack((q2_tilde, np.array([1])))

# Triangulate the new 3D point from the noisy projections
qs = [q1_tilde_homogeneous, q2_tilde_homogeneous]
Ps = [P1, P2]

Q_tilde = triangulate_n(qs, Ps)

# Re-project the triangulated point to the image planes
q1_tilde_reprojected = P1 @ Q_tilde
q2_tilde_reprojected = P2 @ Q_tilde

q1_tilde_reprojected /= q1_tilde_reprojected[2]
q2_tilde_reprojected /= q2_tilde_reprojected[2]

# Compute the reprojection errors
error_q1 = np.linalg.norm(q1_tilde - q1_tilde_reprojected[:2])
error_q2 = np.linalg.norm(q2_tilde - q2_tilde_reprojected[:2])

# Compute the distance between the original point Q and the triangulated point Q_tilde
distance_Q = np.linalg.norm(Q[:3] - Q_tilde[:3])

# Print the results
print("Noisy projection q1_tilde:", q1_tilde.flatten())
print("Noisy projection q2_tilde:", q2_tilde.flatten())
print("Triangulated point Q_tilde:", Q_tilde.flatten())
print(f"The reprojection error in camera 1 is {error_q1:.1f} pixels and it is {error_q2:.2f} pixels in camera 2.")
print(f"The distance between Q and Q_tilde is {distance_Q:.3f}")
