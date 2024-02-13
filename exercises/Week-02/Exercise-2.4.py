import cv2
import numpy as np


def undistortImage(im, K, dist_coeffs):
    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    # Stack and reshape to get homogeneous coordinates
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)

    # Normalize the coordinates
    K_inv = np.linalg.inv(K)
    p_n = K_inv @ p

    # Radial distortion correction
    r = np.sqrt(p_n[0] ** 2 + p_n[1] ** 2)
    r2 = r ** 2
    r4 = r ** 2 * r2
    r6 = r2 * r4
    # Considering only k3, k5, and k7 terms as given
    radial_distortion = 1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r4 + dist_coeffs[2] * r6
    p_n[0] *= radial_distortion
    p_n[1] *= radial_distortion

    # Re-project to pixel coordinates
    p_d = K @ p_n
    # Ensure the third coordinate is 1
    p_d /= p_d[2]

    # Extract x and y coordinates
    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    # Check that the third coordinate is 1
    assert (p_d[2] == 1).all(), 'You did a mistake somewhere'

    # Remap the image to correct for distortion
    im_undistorted = cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)

    return im_undistorted


# Let's test the function with the provided image and calculated camera matrix
# Reload the image
image_path = 'data/gopro_robot.jpg'
im = cv2.imread(image_path)

# Camera matrix K has been calculated previously
K = np.array([
    [875.00544, 0, 960.0],
    [0, 875.00544, 540.0],
    [0, 0, 1]
])

# Distortion coefficients, considering only k3, k5, and k7 as given
dist_coeffs = np.array([-0.245031, 0.071524, -0.00994978])

# Undistort the image
im_undistorted = undistortImage(im, K, dist_coeffs)

# Save the undistorted image
undistorted_image_path = 'data/gopro_robot_undistorted.jpg'
cv2.imwrite(undistorted_image_path, im_undistorted)
