import cv2
import numpy as np
import matplotlib.pyplot as plt


def checkerboard_points(n, m, square_size=1.0):
    """
    Generate 3D points for an n x m checkerboard pattern.

    Parameters:
    - n: Number of internal corners along the width.
    - m: Number of internal corners along the height.
    - square_size: Size of a single square on the checkerboard (default is 1.0).

    Returns:
    - objp: Array of 3D points (n*m, 3).
    """
    # Initialize an array to hold the 3D points
    objp = np.zeros((n * m, 3), np.float32)
    # print(len(objp))
    # Use np.mgrid to create a grid of points
    # np.mgrid[0:n, 0:m] creates two 2D arrays of shape (n, m)
    # .T reshapes these arrays to (n*m, 2) where each row is (x, y) coordinates
    objp[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2)
    # Multiply by square_size to scale the points according to the actual size of the squares
    objp *= square_size

    return objp


def detect_checkerboard(image, pattern_size, criteria):
    """
    Detect checkerboard corners in an image.

    Parameters:
    - image: The input image in which to detect the checkerboard corners.
    - pattern_size: A tuple (number of internal corners in width, number of internal corners in height).
    - criteria: The criteria for refining the corner positions (usually a combination of termination criteria).

    Returns:
    - ret: Boolean indicating whether the checkerboard was found.
    - corners: The refined coordinates of the detected corners if found, otherwise None.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the initial corner positions of the checkerboard pattern
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    # If corners are found, refine their positions
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return ret, corners


if __name__ == '__main__':
    # # Example usage
    # n_corners = 6  # Number of internal corners along the width
    # m_corners = 8  # Number of internal corners along the height
    # square_size = 1.0  # Size of each square
    #
    # # Generate the 3D points
    # checkerboard_3d_points = checkerboard_points(n_corners, m_corners, square_size)
    #
    # # Display the points
    # print("3D points for the checkerboard pattern:")
    # print(checkerboard_3d_points)
    image_path = 'data/chess/3.jpeg'
    image = cv2.imread(image_path)
    pattern_size = (5, 5)
    objp = checkerboard_points(pattern_size[0], pattern_size[1])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Detect the checkerboard corners
    ret, corners = detect_checkerboard(image, pattern_size, criteria)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Check if the checkerboard was found
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print("Checkerboard corners detected.")
        # Draw the detected corners
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)

        # Convert the image from BGR to RGB for displaying with matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image
        plt.imshow(image_rgb)
        plt.title("Detected Checkerboard Corners")
        plt.show()
    else:
        print("Checkerboard corners not detected.")
