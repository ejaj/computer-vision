import numpy as np

P1 = np.array([1, 10, -3, 1])
P2 = np.array([2, -4, 1.1, 2])
P3 = np.array([0, 0, -1, 10])
P4 = np.array([-15, 3, 6, 3])

# Convert to inhomogeneous coordinates (3D)
Q1 = P1[:3] / P1[3]
Q2 = P2[:3] / P2[3]
Q3 = P3[:3] / P3[3]
Q4 = P4[:3] / P4[3]

inhomogeneous_3d = np.array([Q1, Q2, Q3, Q4])
print(inhomogeneous_3d)
