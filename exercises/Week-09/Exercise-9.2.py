import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('data/TwoImageData.npy', allow_pickle=True).item()
print(data)
img1 = data['im1']
img2 = data['im2']

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Use BFMatcher with crossCheck=True
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Visualize the matches
plt.figure(figsize=(12, 6))
plt.imshow(img_matches, cmap='gray')
plt.title('Feature Matches with Cross-Checking')
plt.axis('off')
plt.show()
