import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    return np.hstack(points) / 2


box_points = box3d()
# Reshape the points into a format suitable for plotting
box_points = box_points.reshape(-1, 3, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Create a Poly3DCollection from the corner points and add it to the plot
ax.add_collection3d(Poly3DCollection(box_points, alpha=0.25, edgecolor='k'))
# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Set the aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])
# Show the plot
plt.show()
