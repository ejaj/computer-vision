import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_checkerboard(n, m):
    """Generate n x m checkerboard points on the z=0 plane, centered around the origin."""
    x = np.linspace(-m / 2, m / 2, m)
    y = np.linspace(-n / 2, n / 2, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    return np.vstack([X.flatten(), Y.flatten(), Z.flatten()])


def apply_rotation(points, angle_x):
    """Rotate points around the x-axis by angle_x radians."""
    rot = R.from_euler('x', angle_x).as_matrix()
    return rot @ points


def plot_points(points, title):
    """Plot points in 3D space."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0, :], points[1, :], points[2, :])
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# Generate checkerboard points
n, m = 10, 20  # Dimensions of the checkerboard
Q_omega = generate_checkerboard(n, m)

# Rotation angles
angles = [np.pi / 10, 0, -np.pi / 10]  # +pi/10, 0, -pi/10 radians
titles = ['Qa: +pi/10 rotation', 'Qb: No rotation', 'Qc: -pi/10 rotation']

# Apply rotations and plot results
for angle, title in zip(angles, titles):
    Q_rotated = apply_rotation(Q_omega, angle)
    plot_points(Q_rotated, title)
