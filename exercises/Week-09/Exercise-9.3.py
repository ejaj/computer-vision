import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load data
data = np.load('data/TwoImageData.npy', allow_pickle=True).item()
im1 = data['im1']
im2 = data['im2']
Ftrue = data['R1'] @ data['R2'].T

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(im1, None)
kp2, des2 = orb.detectAndCompute(im2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches
img_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches)
plt.show()

# Extract location of good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)


# Function to estimate fundamental matrix using 8-point algorithm
def Fest_8point(matches):
    A = []
    for (p1, p2) in matches:
        x1, y1 = p1
        x2, y2 = p2
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    return F


# Function to compute Sampson's distance
def SampsonsDistance(F, p1, p2):
    p1_hom = np.append(p1, 1)
    p2_hom = np.append(p2, 1)
    l2 = F @ p1_hom
    l1 = F.T @ p2_hom
    numerator = (p2_hom.T @ F @ p1_hom) ** 2
    denominator = l2[0] ** 2 + l2[1] ** 2 + l1[0] ** 2 + l1[1] ** 2
    return numerator / denominator


# RANSAC algorithm to find the best fundamental matrix
def ransac(matches, pts1, pts2, iterations=200, threshold=3.84 * 3 ** 2):
    best_inliers = []
    best_F = None
    num_matches = len(matches)

    for _ in range(iterations):
        idxs = np.random.choice(num_matches, 8, replace=False)
        sampled_matches = [(pts1[i], pts2[i]) for i in idxs]
        F = Fest_8point(sampled_matches)
        inliers = []
        for i in range(num_matches):
            p1, p2 = pts1[i], pts2[i]
            if SampsonsDistance(F, p1, p2) < threshold:
                inliers.append((p1, p2))
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F
    return best_F, best_inliers


# Run RANSAC
best_F, best_inliers = ransac(matches, pts1, pts2)

# Estimate the final fundamental matrix using all inliers
final_F = Fest_8point(best_inliers)

# Compare the estimated fundamental matrix to the true fundamental matrix
similarity = (final_F * Ftrue).sum() / (np.linalg.norm(final_F) * np.linalg.norm(Ftrue))
print(f'Similarity measure: {similarity}')
