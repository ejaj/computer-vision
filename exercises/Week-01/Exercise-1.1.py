import numpy as np

# Exercise 1.1
p1 = np.array([1, 2, 1])
p2 = np.array([4, 2, 2])
p3 = np.array([6, 4, -1])
p4 = np.array([5, 3, 0.5])
# Convert to inhomogeneous coordinates (2D)
q1 = p1[:2] / p1[2]
q2 = p2[:2] / p2[2]
q3 = p3[:2] / p3[2]
q4 = p4[:2] / p4[2]

inhomogeneous_2d = np.array([q1, q2, q3, q4])
print(inhomogeneous_2d)
