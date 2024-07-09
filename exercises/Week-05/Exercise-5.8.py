import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to generate 3D points on the checkerboard
def checkerboard_points(n, m, square_size=1.0):
    """
    Generate 3D points for an n x m checkerboard.

    Parameters:
    - n: Number of internal corners along the width.
    - m: Number of internal corners along the height.
    - square_size: Size of a single square on the checkerboard.

    Returns:
    - objp: Array of 3D points.
    """
    objp = np.zeros((n * m, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2)
    objp *= square_size
    return objp


# Load images from a directory using OpenCV
def load_images_from_folder_cv2(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


# Resize image for testing
def resize_image(image, scale=0.25):
    return cv2.resize(image, None, fx=scale, fy=scale)


# Detect checkerboard corners
def detect_checkerboard(image, pattern_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
    return ret, corners


# Visualize checkerboard detection
def visualize_checkerboard_detection(image, corners, pattern_size):
    vis_image = image.copy()
    cv2.drawChessboardCorners(vis_image, pattern_size, corners, True)
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.show()


# Calibration criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Load images
folder_path = 'data/chess'
images, filenames = load_images_from_folder_cv2(folder_path)

# Checkerboard pattern size
pattern_size = (5, 5)

# Prepare object points
objp = checkerboard_points(pattern_size[0], pattern_size[1])

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Detect checkerboards in all images
successful_images = []
successful_filenames = []
for img, fname in zip(images, filenames):
    ret, corners = detect_checkerboard(img, pattern_size)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        successful_images.append(img)
        successful_filenames.append(fname)
        print(f"Checkerboard detected in image: {fname}")
        visualize_checkerboard_detection(img, corners, pattern_size)
    else:
        print(f"Checkerboard not detected in image: {fname}")

print(f"Number of images with detected checkerboards: {len(successful_images)}")

# Camera calibration
if len(objpoints) > 0:
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, successful_images[0].shape[1::-1], None,
                                                     None,
                                                     flags=cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 +
                                                           cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 +
                                                           cv2.CALIB_ZERO_TANGENT_DIST)

    print("Camera matrix (K):")
    print(K)

    # Check if the principal point is approximately in the center
    image_center = (successful_images[0].shape[1] / 2, successful_images[0].shape[0] / 2)
    principal_point = (K[0, 2], K[1, 2])
    print(f"Image center: {image_center}")
    print(f"Principal point: {principal_point}")

    if abs(image_center[0] - principal_point[0]) < 50 and abs(image_center[1] - principal_point[1]) < 50:
        print("The principal point is approximately in the center of the image.")
    else:
        print("The principal point is not in the center of the image.")
else:
    print("Not enough valid images for calibration.")
