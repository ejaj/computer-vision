import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

# Scattered data points and their values
points = np.array([[1, 1], [3, 2], [5, 4], [6, 1]])  # Locations (x, y)
values = np.array([20, 25, 30, 22])  # Temperature values

# Create a Delaunay triangulation
tri = Delaunay(points)

# Create a linear interpolator
interpolator = LinearNDInterpolator(points, values)

# Generate a grid of points for visualization
x = np.linspace(0, 7, 100)
y = np.linspace(0, 5, 100)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Interpolate values on the grid
interpolated_values = interpolator(grid_points).reshape(xx.shape)

# Plotting the "before" (raw data) and "after" (interpolated data) side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# "Before" plot: raw data points
axes[0].scatter(points[:, 0], points[:, 1], color='red', s=100, label='Data Points')
for i, val in enumerate(values):
    axes[0].text(points[i, 0], points[i, 1], f"{val}°C", color="black", fontsize=10, ha='center', va='center')
axes[0].set_title("Before: Raw Scattered Data")
axes[0].set_xlabel("X Coordinate")
axes[0].set_ylabel("Y Coordinate")
axes[0].legend()
axes[0].grid(True)
axes[0].set_xlim(0, 7)
axes[0].set_ylim(0, 5)

# "After" plot: interpolated surface
contour = axes[1].contourf(xx, yy, interpolated_values, levels=100, cmap='coolwarm')
plt.colorbar(contour, ax=axes[1], label='Temperature (°C)')
axes[1].scatter(points[:, 0], points[:, 1], color='black', label='Data Points', edgecolor='white')
for i, val in enumerate(values):
    axes[1].text(points[i, 0], points[i, 1], f"{val}°C", color="white", fontsize=8, ha='center', va='center')
axes[1].set_title("After: Interpolated Data")
axes[1].set_xlabel("X Coordinate")
axes[1].set_ylabel("Y Coordinate")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
