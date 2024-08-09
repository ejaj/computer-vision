import cv2
import numpy as np
from matplotlib import pyplot as plt

K = np.loadtxt('data/K.txt')
im0 = cv2.imread('data/sequence/000001.png')
im1 = cv2.imread('data/sequence/000002.png')
im2 = cv2.imread('data/sequence/000003.png')

sift = cv2.SIFT_create(nfeatures=2000)

kp0, des0 = sift.detectAndCompute(im0, None)
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

kp0_np = np.array([k.pt for k in kp0])
kp1_np = np.array([k.pt for k in kp1])
kp2_np = np.array([k.pt for k in kp2])

bf = cv2.BFMatcher()

matches01 = bf.knnMatch(des0, des1, k=2)

# Apply ratio test as per Lowe's paper
good_matches01 = []
for m, n in matches01:
    if m.distance < 0.75 * n.distance:
        good_matches01.append(m)

# Match between im1 and im2
matches12 = bf.knnMatch(des1, des2, k=2)

# Apply ratio test as per Lowe's paper
good_matches12 = []
for m, n in matches12:
    if m.distance < 0.75 * n.distance:
        good_matches12.append(m)

matches01_np = np.array([(m.queryIdx, m.trainIdx) for m in good_matches01])
matches12_np = np.array([(m.queryIdx, m.trainIdx) for m in good_matches12])

# Step 1: Estimate the Essential Matrix
E, mask = cv2.findEssentialMat(kp0_np[matches01_np[:, 0]], kp1_np[matches01_np[:, 1]], K, method=cv2.RANSAC, prob=0.999,
                               threshold=1.0)
# Step 2: Decompose the Essential Matrix to find the relative pose
_, R1, t1, mask_pose = cv2.recoverPose(E, kp0_np[matches01_np[:, 0]], kp1_np[matches01_np[:, 1]], K)
# Step 3: Filter out non-inlier matches
inlier_matches01_np = matches01_np[mask_pose.ravel() == 1]
# Optional: Visualize the inlier matches
inlier_matches = [good_matches01[i] for i in range(len(good_matches01)) if mask_pose[i]]
im_matches = cv2.drawMatches(im0, kp0, im1, kp1, inlier_matches, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.figure(figsize=(20, 10))
plt.title('Inlier Matches')
plt.imshow(im_matches)
plt.show()
