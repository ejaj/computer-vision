import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools as it


def checkerboard_points(n, m, square_size=1.0):
    objp = np.zeros((n * m, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2)
    objp *= square_size
    return objp


# Function to generate 3D points in a box shape
def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    points = np.hstack(points)
    # Convert points to homogeneous coordinates by adding a row of ones
    points_homogeneous = np.vstack((points, np.ones(points.shape[1])))
    return points_homogeneous / 2


# Generate the 3D points
Q = 2 * box3d() + 1


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
def detect_checkerboard(image, pattern_size, criteria):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners


# Visualize checkerboard detection
def visualize_checkerboard_detection(image, corners, reprojected_corners, pattern_size):
    vis_image = image.copy()
    cv2.drawChessboardCorners(vis_image, pattern_size, corners, True)
    for point in reprojected_corners:
        cv2.circle(vis_image, (int(point[0][0]), int(point[0][1])), 3, (0, 0, 255),
                   -1)  # Red color for reprojected points
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
    ret, corners = detect_checkerboard(img, pattern_size, criteria)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        successful_images.append(img)
        successful_filenames.append(fname)
        print(f"Checkerboard detected in image: {fname}")
    else:
        print(f"Checkerboard not detected in image: {fname}")

print(f"Number of images with detected checkerboards: {len(successful_images)}")

# Camera calibration with first order distortion coefficient (k1)
if len(objpoints) > 0:
    flags = cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + \
            cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6 + cv2.CALIB_ZERO_TANGENT_DIST

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, successful_images[0].shape[1::-1], None,
                                                     None, flags=flags)

    print("Camera matrix (K) with k1 distortion coefficient:")
    print(K)
    print("Distortion coefficients:")
    print(dist)

    # Check if the principal point is approximately in the center
    image_center = (successful_images[0].shape[1] / 2, successful_images[0].shape[0] / 2)
    principal_point = (K[0, 2], K[1, 2])
    print(f"Image center: {image_center}")
    print(f"Principal point: {principal_point}")

    if abs(image_center[0] - principal_point[0]) < 50 and abs(image_center[1] - principal_point[1]) < 50:
        print("The principal point is approximately in the center of the image.")
    else:
        print("The principal point is not in the center of the image.")

    # Reproject the checkerboard corners
    total_error = 0
    max_error = 0
    worst_image_idx = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        if error > max_error:
            max_error = error
            worst_image_idx = i

    mean_error = total_error / len(objpoints)
    print(f"Mean reprojection error with k1: {mean_error:.2f} pixels")
    print(
        f"Highest reprojection error with k1: {max_error:.2f} pixels in image: {successful_filenames[worst_image_idx]}")

    # Visualize the worst image with detected and reprojected corners
    worst_image = successful_images[worst_image_idx]
    detected_corners = imgpoints[worst_image_idx]
    reprojected_corners, _ = cv2.projectPoints(objpoints[worst_image_idx], rvecs[worst_image_idx],
                                               tvecs[worst_image_idx], K, dist)
    visualize_checkerboard_detection(worst_image, detected_corners, reprojected_corners, pattern_size)

    # Use the estimated R and t to project the box3D points onto the image
    Q = Q[:3].T  # Remove the homogeneous coordinate and transpose to match projectPoints input
    R, _ = cv2.Rodrigues(rvecs[worst_image_idx])
    t = tvecs[worst_image_idx]
    projected_points, _ = cv2.projectPoints(Q, rvecs[worst_image_idx], tvecs[worst_image_idx], K, dist)

    # Visualize the projected 3D box points on the image
    image_with_box = worst_image.copy()
    for point in projected_points:
        cv2.circle(image_with_box, (int(point[0][0]), int(point[0][1])), 3, (255, 0, 0),
                   -1)  # Blue color for box points
    plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))
    plt.show()

else:
    print("Not enough valid images for calibration.")
