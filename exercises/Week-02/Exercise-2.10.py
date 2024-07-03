import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_points(image, num_points):
    plt.imshow(image)
    print(f"Please select {num_points} points. Right click to stop selection early.")
    points = plt.ginput(num_points, timeout=-1)
    plt.close()
    if len(points) != num_points:
        raise ValueError(f"Expected {num_points} points, but got {len(points)}.")
    return np.array(points).T


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


def click_and_map(imageA, imageB, H):
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        point = np.array([[ix], [iy]])
        mapped_point = map_points(H, point)
        print(f"Clicked point in image A: ({ix}, {iy})")
        print(f"Mapped point in image B: ({mapped_point[0, 0]}, {mapped_point[1, 0]})")

        # Show image B with the mapped point
        fig, ax = plt.subplots()
        ax.imshow(imageB)
        ax.scatter(mapped_point[0, 0], mapped_point[1, 0], c='r', marker='x')
        plt.show()

    fig, ax = plt.subplots()
    ax.imshow(imageA)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


# Load the images
imageA_path = 'data/a.jpeg'
imageB_path = 'data/b.jpeg'
imageA = cv2.cvtColor(cv2.imread(imageA_path), cv2.COLOR_BGR2RGB)
imageB = cv2.cvtColor(cv2.imread(imageB_path), cv2.COLOR_BGR2RGB)

# Get the points from both images
num_points = 4
print("Select points in image A")
pointsA = get_points(imageA, num_points)
print("Select points in image B")
pointsB = get_points(imageB, num_points)

# Estimate the homography matrix
H = estimate_homography(pointsB, pointsA)

# Test by clicking on image A and mapping to image B
click_and_map(imageA, imageB, H)
