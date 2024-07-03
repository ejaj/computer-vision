import numpy as np
import cv2


def warpImage(im, H):
    imWarp = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
    return imWarp


# Global variables for storing points
pointsA = []
pointsB = []


# Mouse callback function for selecting points
def select_points(event, x, y, flags, param):
    global pointsA, pointsB, selecting_A

    if event == cv2.EVENT_LBUTTONDOWN:
        if selecting_A:
            if len(pointsA) < 4:
                pointsA.append((x, y))
                cv2.circle(imageA_display, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Image A", imageA_display)
        else:
            if len(pointsB) < 4:
                pointsB.append((x, y))
                cv2.circle(imageB_display, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Image B", imageB_display)


def estimate_homography(q1, q2):
    """ Estimate the homography matrix H that maps q2 to q1 using the DLT algorithm. """
    A = []
    for i in range(q1.shape[1]):
        x, y = q2[0, i], q2[1, i]
        u, v = q1[0, i], q1[1, i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H_est = Vt[-1].reshape(3, 3)
    return H_est


def select_points_in_image(image, window_name, points):
    global selecting_A, imageA_display, imageB_display
    selecting_A = (window_name == "Image A")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_points)

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) >= 4:
            break

    cv2.destroyAllWindows()


# Load the images
imageA_path = 'data/a.jpeg'
imageB_path = 'data/b.jpeg'
imageA = cv2.imread(imageA_path)
imageB = cv2.imread(imageB_path)
imageA_display = imageA.copy()
imageB_display = imageB.copy()

# Select points in image A
print("Select 4 points in image A")
select_points_in_image(imageA_display, "Image A", pointsA)

# Select points in image B
print("Select 4 points in image B")
select_points_in_image(imageB_display, "Image B", pointsB)

# Convert points to homogeneous coordinates
pointsA = np.array(pointsA).T
pointsB = np.array(pointsB).T

# Debugging: Print the points to verify
print("Points in image A (pointsA):")
print(pointsA)
print("Points in image B (pointsB):")
print(pointsB)

# Ensure points are in the correct shape
assert pointsA.shape[0] == 2 and pointsA.shape[1] == 4, "pointsA shape is incorrect"
assert pointsB.shape[0] == 2 and pointsB.shape[1] == 4, "pointsB shape is incorrect"

# Estimate the homography matrix
H = estimate_homography(pointsB, pointsA)

# Warp image B to align with image A
imageB_warped = warpImage(imageB, H)

# Create an overlay of the two images
overlay = cv2.addWeighted(imageA, 0.5, imageB_warped, 0.5, 0)

# Display the results
cv2.imshow("Image A", imageA)
cv2.imshow("Warped Image B", imageB_warped)
cv2.imshow("Overlay", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
