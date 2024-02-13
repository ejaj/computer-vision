import numpy as np

p2a_cartesian = np.array([1, 1])
p2b_cartesian = np.array([0, 3])
p2c_cartesian = np.array([2, 3])
p2d_cartesian = np.array([2, 4])

# Convert to homogeneous coordinates by adding a third coordinate with value 1
p2a = np.append(p2a_cartesian, 1)
p2b = np.append(p2b_cartesian, 1)
p2c = np.append(p2c_cartesian, 1)
p2d = np.append(p2d_cartesian, 1)

# Homography matrix
H = np.array([
    [-2, 0, 1],
    [1, -2, 0],
    [0, 0, 3]
])

# Apply the homography to the points
q2a_h = H @ p2a
q2b_h = H @ p2b
q2c_h = H @ p2c
q2d_h = H @ p2d

# Normalize to get the final homogeneous coordinates
q2a_h_normalized = q2a_h / q2a_h[2]
q2b_h_normalized = q2b_h / q2b_h[2]
q2c_h_normalized = q2c_h / q2c_h[2]
q2d_h_normalized = q2d_h / q2d_h[2]

print(q2a_h_normalized)
print(q2b_h_normalized)
print(q2c_h_normalized)
print(q2d_h_normalized)
