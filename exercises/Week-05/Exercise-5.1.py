import numpy as np

# Define intrinsic matrix K1 = K2
K = np.array([
    [700, 0, 600],
    [0, 700, 400],
    [0, 0, 1]
])

# Define rotation matrix R1 = R2 = I (identity matrix)
R = np.eye(3)

# Define translation vectors t1 and t2
t1 = np.array([0, 0, 1]).reshape((3, 1))
t2 = np.array([0, 0, 20]).reshape((3, 1))

# Construct the projection matrices P1 and P2
P1 = K @ np.hstack((R, t1))
P2 = K @ np.hstack((R, t2))
# Define the 3D point Q
Q = np.array([1, 1, 0, 1]).reshape((4, 1))

# Compute the projections q1 and q2
q1_homogeneous = P1 @ Q
q2_homogeneous = P2 @ Q

# Convert homogeneous coordinates to non-homogeneous coordinates
q1 = q1_homogeneous[:2] / q1_homogeneous[2]
q2 = q2_homogeneous[:2] / q2_homogeneous[2]

# Print the results
print("Projection matrix P1:")
print(P1)

print("\nProjection matrix P2:")
print(P2)

print("\nProjection q1 (Camera 1):")
print(q1)

print("\nProjection q2 (Camera 2):")
print(q2)
