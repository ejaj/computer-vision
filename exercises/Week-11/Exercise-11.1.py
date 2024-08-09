import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load the camera matrix
K = np.loadtxt('data/K.txt')
# print(K)
# Step 2: Load the images
im0 = cv2.imread('data/sequence/000001.png')
im1 = cv2.imread('data/sequence/000002.png')
im2 = cv2.imread('data/sequence/000003.png')

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

# Step 6: Convert the matches to numpy arrays of indices
matches01_np = np.array([(m.queryIdx, m.trainIdx) for m in good_matches01])
matches12_np = np.array([(m.queryIdx, m.trainIdx) for m in good_matches12])

# Plotting the results for visualization

# Draw matches between im0 and im1
im_matches01 = cv2.drawMatches(im0, kp0, im1, kp1, good_matches01, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw matches between im1 and im2
im_matches12 = cv2.drawMatches(im1, kp1, im2, kp2, good_matches12, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the images
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.title('Matches between 000001.png and 000002.png')
plt.imshow(im_matches01)

plt.subplot(1, 2, 2)
plt.title('Matches between 000002.png and 000003.png')
plt.imshow(im_matches12)

plt.show()
