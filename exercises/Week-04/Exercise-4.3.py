import numpy as np


def checkerboard_points(n, m):
    # Create meshgrid for indices
    j, i = np.meshgrid(range(m), range(n))
    # Calculate coordinates centered around (0, 0)
    x = i - (n - 1) / 2
    y = j - (m - 1) / 2
    z = np.zeros_like(x)  # z-coordinates are zero

    # Stack and reshape to form a 3 x (n*m) matrix
    points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    return points


# Example usage
n = 10  # Number of rows
m = 20  # Number of columns
points = checkerboard_points(n, m)
print("Checkerboard points (3 x n*m matrix):")
print(points)
