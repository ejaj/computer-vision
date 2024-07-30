import random
import matplotlib.pyplot as plt


def draw_two_points(points):
    """
    Randomly draw two distinct points from a list of 2D points.

    :param points: List of tuples (x, y) representing the points.
    :return: A tuple of two randomly selected distinct points.
    """
    if len(points) < 2:
        raise ValueError("At least two points are required to draw two distinct points.")

    return random.sample(points, 2)


# Generate random points
points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]

# Draw two points
selected_points = draw_two_points(points)

# Unzip the points for plotting
x_all, y_all = zip(*points)
x_selected, y_selected = zip(*selected_points)

# Plot all points
plt.scatter(x_all, y_all, color='blue', label='All Points')

# Highlight the selected points
plt.scatter(x_selected, y_selected, color='red', s=100, edgecolors='k', label='Selected Points')

# Add labels and legend
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Randomly Drawn Points')
plt.legend()

# Show plot
plt.show()
