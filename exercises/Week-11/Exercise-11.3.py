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

# Matches between im0 and im1
im_matches01 = cv2.drawMatches(im0, kp0, im1, kp1, [good_matches01[i] for i in idx01], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Matches between im1 and im2
im_matches12 = cv2.drawMatches(im1, kp1, im2, kp2, [good_matches12[i] for i in idx12], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the images
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.title('Inlier Matches between im0 and im1')
plt.imshow(im_matches01)

plt.subplot(1, 2, 2)
plt.title('Inlier Matches between im1 and im2')
plt.imshow(im_matches12)

plt.show()
