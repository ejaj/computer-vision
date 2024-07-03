import cv2
import numpy as np

# Load the image
image_path = 'data/gopro_robot.jpg'
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Given parameters
focal_length_factor = 0.455732
f = focal_length_factor * image_width

# Reasonable guess for principal point (center of the image)
delta_x = image_width / 2
delta_y = image_height / 2

# Skew coefficients
alpha = 0  # Usually 0
beta = 0  # Usually 0

# Distortion coefficients
k3 = -0.245031
k5 = 0.071524
k7 = -0.00994978

distCoeffs = [k3, k5, k7]

# Intrinsic camera matrix K
K = np.array([
    [f, alpha, delta_x],
    [0, f, delta_y],
    [0, 0, 1]
])

print("Intrinsic camera matrix K:")
print(K)
print("Distortion coefficients:")
print(distCoeffs)
