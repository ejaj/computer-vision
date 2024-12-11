import numpy as np
import matplotlib.pyplot as plt
# Fit a function using interpolation techniques
from scipy.interpolate import Rbf


# Define the grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Generate a random 2D function
np.random.seed(0)
num_components = 3  # Number of components in the function
function = np.zeros_like(X)

for _ in range(num_components):
    amplitude = np.random.uniform(0.5, 2)
    freq_x = np.random.uniform(0.5, 1.5)
    freq_y = np.random.uniform(0.5, 1.5)
    function += amplitude * np.sin(freq_x * X) * np.sin(freq_y * Y)

plt.figure(figsize=(6, 5))
plt.title("Original Function")
plt.contourf(X, Y, function, levels=50, cmap='viridis')
plt.colorbar(label='Function Value')
plt.show()
# Sample the function at random locations
# Random sampling
num_samples = 50
sample_indices = np.random.choice(function.size, num_samples, replace=False)
sample_x = X.ravel()[sample_indices]
sample_y = Y.ravel()[sample_indices]
sample_values = function.ravel()[sample_indices]

plt.figure(figsize=(6, 5))
plt.title("Sampled Data Points")
plt.scatter(sample_x, sample_y, c=sample_values, cmap='viridis', s=50)
plt.colorbar(label='Function Value')
plt.show()



# RBF interpolation
rbf_interpolator = Rbf(sample_x, sample_y, sample_values, function='multiquadric', smooth=0.1)
interpolated_values = rbf_interpolator(X, Y)

plt.figure(figsize=(6, 5))
plt.title("Interpolated Function")
plt.contourf(X, Y, interpolated_values, levels=50, cmap='viridis')
plt.colorbar(label='Function Value')
plt.show()

# Measure the fitting error

# Error computation
true_values = function.ravel()
interp_values = rbf_interpolator(X.ravel(), Y.ravel())
error = np.abs(true_values - interp_values)

print(f"Mean Absolute Error: {np.mean(error):.4f}")
print(f"Maximum Error: {np.max(error):.4f}")

# Repeat with a new set of sample points

# New random sample
sample_indices_new = np.random.choice(function.size, num_samples, replace=False)
sample_x_new = X.ravel()[sample_indices_new]
sample_y_new = Y.ravel()[sample_indices_new]
sample_values_new = function.ravel()[sample_indices_new]

# New interpolation
rbf_interpolator_new = Rbf(sample_x_new, sample_y_new, sample_values_new, function='multiquadric', smooth=0.1)
interpolated_values_new = rbf_interpolator_new(X, Y)

# Compute new error
interp_values_new = rbf_interpolator_new(X.ravel(), Y.ravel())
error_new = np.abs(true_values - interp_values_new)

print(f"New Mean Absolute Error: {np.mean(error_new):.4f}")
print(f"New Maximum Error: {np.max(error_new):.4f}")
