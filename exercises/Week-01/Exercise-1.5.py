# Define the line vectors from the exercise
import numpy as np

l0 = np.array([1, 1, -1])
l1 = np.array([-1, 1, -3])

# Calculate the cross product to find the intersection point q
q = np.cross(l0, l1)
print(q)
