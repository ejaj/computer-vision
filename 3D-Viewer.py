import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Primitive3DViewer:
    def __init__(self):
        self.points = []  # List of 3D points
        self.lines = []   # List of lines (pairs of points)
        self.polygons = []  # List of polygons (list of points)

    def add_point(self, point):
        """Add a single 3D point (x, y, z)."""
        self.points.append(point)

    def add_line(self, p1, p2):
        """Add a line defined by two points."""
        self.lines.append((p1, p2))

    def add_polygon(self, vertices):
        """Add a polygon defined by a list of vertices."""
        self.polygons.append(vertices)

class Viewer3D(Primitive3DViewer):
    def __init__(self):
        super().__init__()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def display(self):
        """Render the 3D scene."""
        self.ax.clear()

        # Plot points
        if self.points:
            points = np.array(self.points)
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', label='Points')

        # Plot lines
        for p1, p2 in self.lines:
            x_vals = [p1[0], p2[0]]
            y_vals = [p1[1], p2[1]]
            z_vals = [p1[2], p2[2]]
            self.ax.plot(x_vals, y_vals, z_vals, c='b', label='Lines')

        # Plot polygons
        for vertices in self.polygons:
            vertices = np.array(vertices)
            self.ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], alpha=0.5, color='g')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.legend()
        plt.show()

viewer = Viewer3D()

# Add primitives
viewer.add_point((0, 0, 0))
viewer.add_point((1, 1, 1))
viewer.add_line((0, 0, 0), (1, 1, 1))
viewer.add_polygon([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])

# Display the scene
viewer.display()