import numpy as np
import cv2

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


def map_points(H, points):
    points_homogeneous = np.vstack((points, np.ones((1, points.shape[1]))))
    mapped_points_homogeneous = H @ points_homogeneous
    mapped_points = mapped_points_homogeneous[:-1] / mapped_points_homogeneous[-1]
    return mapped_points


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


# Test by clicking on image A and mapping to image B
def click_and_map(imageA, imageB, H):
    def onclick(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = np.array([[x], [y]])
            mapped_point = map_points(H, point)
            print(f"Clicked point in image A: ({x}, {y})")
            print(f"Mapped point in image B: ({mapped_point[0, 0]}, {mapped_point[1, 0]})")

            # Show image B with the mapped point
            img_copy = imageB.copy()
            cv2.circle(img_copy, (int(mapped_point[0, 0]), int(mapped_point[1, 0])), 5, (0, 0, 255), -1)
            cv2.imshow("Mapped Point in Image B", img_copy)

    cv2.namedWindow("Image A Click")
    cv2.setMouseCallback("Image A Click", onclick)

    while True:
        cv2.imshow("Image A Click", imageA)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


click_and_map(imageA, imageB, H)
