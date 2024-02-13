import numpy as np


def Pi(points_homogeneous):
    # Convert from homogeneous to inhomogeneous coordinates
    inhomogeneous_points = points_homogeneous[:-1] / points_homogeneous[-1]
    return inhomogeneous_points


def PiInv(points_inhomogeneous):
    # Convert from inhomogeneous to homogeneous coordinates
    num_points = points_inhomogeneous.shape[1]
    ones_column = np.ones((1, num_points))
    homogeneous_points = np.vstack((points_inhomogeneous, ones_column))
    return homogeneous_points


# Example usage:
# Creating a Numpy array of homogeneous points
homogeneous_points = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0]])

# Converting from homogeneous to inhomogeneous coordinates
inhomogeneous_result = Pi(homogeneous_points)
print("Inhomogeneous Result:")
print(inhomogeneous_result)

# Converting from inhomogeneous back to homogeneous coordinates
homogeneous_result = PiInv(inhomogeneous_result)
print("\nHomogeneous Result:")
print(homogeneous_result)
