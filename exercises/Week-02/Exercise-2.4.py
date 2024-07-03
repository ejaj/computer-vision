import numpy as np
import cv2


def undistortImage(im, K, distCoeffs):
    # Generate a mesh grid for the image coordinates
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)

    # Compute the normalized coordinates
    K_inv = np.linalg.inv(K)
    p_normalized = K_inv @ p

    # Compute radial distances squared
    x_n = p_normalized[0, :]
    y_n = p_normalized[1, :]
    r2 = x_n ** 2 + y_n ** 2

    # Compute the distortion
    k1, k2, k3 = distCoeffs
    radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

    # Apply the distortion to the normalized coordinates
    x_distorted = x_n * radial_distortion
    y_distorted = y_n * radial_distortion

    # Convert distorted normalized coordinates back to pixel coordinates
    p_d = np.vstack((x_distorted, y_distorted, np.ones_like(x_distorted)))
    q = K @ p_d

    # Normalize homogeneous coordinates
    p_d = q / q[2]

    # Reshape distorted coordinates back to the image shape
    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)

    assert (p_d[2] == 1).all(), 'You did a mistake somewhere'

    # Undistort the image using the distorted coordinates
    im_undistorted = cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)

    return im_undistorted


# Load the image
image_path = 'data/gopro_robot.jpg'
image = cv2.imread(image_path)

# Given parameters for the camera matrix and distortion coefficients
focal_length_factor = 0.455732
image_height, image_width = image.shape[:2]
f = focal_length_factor * image_width

# Principal point (center of the image)
delta_x = image_width / 2
delta_y = image_height / 2

# Skew coefficients
alpha = 0
beta = 0

# Distortion coefficients
distCoeffs = [-0.245031, 0.071524, -0.00994978]

# Intrinsic camera matrix K
K = np.array([
    [f, alpha, delta_x],
    [0, f, delta_y],
    [0, 0, 1]
])

# Undistort the image
im_undistorted = undistortImage(image, K, distCoeffs)

# Display the original and undistorted images
cv2.imshow('Original Image', image)
cv2.imshow('Undistorted Image', im_undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
