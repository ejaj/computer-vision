import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


# Step 1: Load all images from the folder
folder = 'data/sequence/'  # Replace with your folder path
images = load_images_from_folder(folder)
K = np.loadtxt('data/K.txt')  # Load the camera matrix

# Initialize storage for results
Rs = []  # Rotation matrices
ts = []  # Translation vectors
points_3d_list = []  # 3D points

# Step 2: Iterate through the images in groups of three
sift = cv2.SIFT_create(nfeatures=2000)
bf = cv2.BFMatcher()

for i in range(2, len(images)):
    im0, im1, im2 = images[i - 2], images[i - 1], images[i]

    # Step 3: Detect and compute SIFT keypoints and descriptors
    kp0, des0 = sift.detectAndCompute(im0, None)
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    kp0_np = np.array([k.pt for k in kp0])
    kp1_np = np.array([k.pt for k in kp1])
    kp2_np = np.array([k.pt for k in kp2])

    # Step 4: Match features between images
    matches01 = bf.knnMatch(des0, des1, k=2)
    matches12 = bf.knnMatch(des1, des2, k=2)

    good_matches01 = [m for m, n in matches01 if m.distance < 0.85 * n.distance]
    good_matches12 = [m for m, n in matches12 if m.distance < 0.85 * n.distance]

    matches01_np = np.array([(m.queryIdx, m.trainIdx) for m in good_matches01])
    matches12_np = np.array([(m.queryIdx, m.trainIdx) for m in good_matches12])

    # Step 5: Find common matches across the three images
    _, idx01, idx12 = np.intersect1d(matches01_np[:, 1], matches12_np[:, 0], return_indices=True)
    points0 = kp0_np[matches01_np[idx01, 0]]
    points1 = kp1_np[matches01_np[idx01, 1]]
    points2 = kp2_np[matches12_np[idx12, 1]]

    # Step 6: Triangulate 3D points using image 0 and image 1
    E, mask_pose = cv2.findEssentialMat(points0, points1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R1, t1, _ = cv2.recoverPose(E, points0, points1, K, mask=mask_pose)

    points0 = points0[mask_pose.ravel() == 1]
    points1 = points1[mask_pose.ravel() == 1]
    points2 = points2[mask_pose.ravel() == 1]

    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = np.hstack((R1, t1))

    P0 = K @ P0
    P1 = K @ P1

    points_4d_hom = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)

    # Store 3D points
    points_3d_list.append(points_3d)

    # Step 7: Estimate pose of im2
    distCoeffs = np.zeros(5)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points2, K, distCoeffs)

    if success:
        R2, _ = cv2.Rodrigues(rvec)
        Rs.append(R2)
        ts.append(tvec)
    else:
        print(f"Pose estimation failed for image {i + 1}")

# Step 8: Visualize the 3D points and camera positions
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Visualize 3D points
for points_3d in points_3d_list:
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')

# Visualize camera positions
ax.scatter(0, 0, 0, c='blue', marker='^', label='Camera 1')
camera_position = np.zeros(3)
for R, t in zip(Rs, ts):
    camera_position += (-R.T @ t).flatten()  # Flatten to ensure it's a vector
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='green', marker='^')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
