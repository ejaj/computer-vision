import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the camera matrix
K = np.loadtxt('data/K.txt')

# Step 2: Load the images
im0 = cv2.imread('data/sequence/000001.png', cv2.IMREAD_GRAYSCALE)
im1 = cv2.imread('data/sequence/000002.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('data/sequence/000003.png', cv2.IMREAD_GRAYSCALE)

# Step 3: Initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=2000)

# Step 4: Find keypoints and descriptors
kp0, des0 = sift.detectAndCompute(im0, None)
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# Convert keypoints to numpy arrays of 2D points
kp0_np = np.array([k.pt for k in kp0])
kp1_np = np.array([k.pt for k in kp1])
kp2_np = np.array([k.pt for k in kp2])

# Step 5: Match SIFT features between images
bf = cv2.BFMatcher()

# Match between im0 and im1
matches01 = bf.knnMatch(des0, des1, k=2)
# Match between im1 and im2
matches12 = bf.knnMatch(des1, des2, k=2)

# Apply a less strict ratio test
good_matches01 = []
for m, n in matches01:
    if m.distance < 0.85 * n.distance:  # Increased to 0.85
        good_matches01.append(m)

good_matches12 = []
for m, n in matches12:
    if m.distance < 0.85 * n.distance:  # Increased to 0.85
        good_matches12.append(m)

# Debugging: Check the number of good matches
print(f"Number of good matches between im0 and im1: {len(good_matches01)}")
print(f"Number of good matches between im1 and im2: {len(good_matches12)}")

# Convert the matches to numpy arrays of indices
matches01_np = np.array([(m.queryIdx, m.trainIdx) for m in good_matches01])
matches12_np = np.array([(m.queryIdx, m.trainIdx) for m in good_matches12])

# Debugging: Print the matches to check overlap
print("Sample of matches between im0 and im1:")
print(matches01_np[:10])

print("Sample of matches between im1 and im2:")
print(matches12_np[:10])

# Step 6: Perform intersection to find common matches
_, idx01, idx12 = np.intersect1d(matches01_np[:, 1], matches12_np[:, 0], return_indices=True)

# Debugging: Check intersection results
print(f"Number of common matches found: {len(idx01)}")

# Step 7: Extract the corresponding 2D points from the keypoint arrays
points0 = kp0_np[matches01_np[idx01, 0]]  # Points from image 0
points1 = kp1_np[matches01_np[idx01, 1]]  # Points from image 1 (shared)
points2 = kp2_np[matches12_np[idx12, 1]]  # Points from image 2

# Debugging: Print out the number of points found in each image
print(f"Number of points in im0 after intersection: {len(points0)}")
print(f"Number of points in im1 after intersection: {len(points1)}")
print(f"Number of points in im2 after intersection: {len(points2)}")

# Step 8: Estimate the Essential Matrix between im0 and im1
E, mask_pose = cv2.findEssentialMat(points0, points1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Step 9: Decompose the Essential Matrix to find the relative pose between im0 and im1
_, R1, t1, _ = cv2.recoverPose(E, points0, points1, K, mask=mask_pose)

# Filter points based on the mask from cv2.recoverPose
points0 = points0[mask_pose.ravel() == 1]
points1 = points1[mask_pose.ravel() == 1]
points2 = points2[mask_pose.ravel() == 1]  # Adjust points2 as well to keep consistency

# Step 10: Triangulate points in 3D using image 0 and image 1
points0_h = cv2.convertPointsToHomogeneous(points0).reshape(-1, 3)
points1_h = cv2.convertPointsToHomogeneous(points1).reshape(-1, 3)

# Projection matrices for image 0 and image 1
P0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for im0 (assuming the first camera is at origin)
P1 = np.hstack((R1, t1))  # Projection matrix for im1

# Multiply by the camera matrix K
P0 = K @ P0
P1 = K @ P1

# Triangulate the 3D points
points_4d_hom = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)

# Step 11: Estimate pose of im2 using the 3D points and 2D points from im2
distCoeffs = np.zeros(5)  # Assuming no lens distortion
success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points2, K, distCoeffs)

if success:
    # Convert rotation vector to rotation matrix
    R2, _ = cv2.Rodrigues(rvec)

    # Step 12: Visualize the 3D points and camera positions
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_3d[inliers.flatten(), 0], points_3d[inliers.flatten(), 1], points_3d[inliers.flatten(), 2], c='r',
               marker='o')

    # Plot the camera positions
    # Camera 0 is at the origin (0, 0, 0)
    ax.scatter(0, 0, 0, c='blue', marker='^', label='Camera 0')

    # Camera 1 position can be found from R1 and t1
    camera1_position = -R1.T @ t1
    ax.scatter(camera1_position[0], camera1_position[1], camera1_position[2], c='green', marker='^', label='Camera 1')

    # Camera 2 position can be found from R2 and tvec
    camera2_position = -R2.T @ tvec
    ax.scatter(camera2_position[0], camera2_position[1], camera2_position[2], c='purple', marker='^', label='Camera 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
else:
    print("Pose estimation for im2 failed.")
