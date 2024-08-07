import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

img_1 = Image.open('data/1.png')
img_2 = Image.open('data/2.png')

img_1_cv = cv2.cvtColor(np.asarray(img_1), cv2.COLOR_RGB2BGR)
img_2_cv = cv2.cvtColor(np.asarray(img_2), cv2.COLOR_RGB2BGR)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img_1_cv, None)
keypoints2, descriptors2 = sift.detectAndCompute(img_2_cv, None)

img_1_with_keypoints = cv2.drawKeypoints(
    img_1_cv,
    keypoints1, None
)
img_2_with_keypoints = cv2.drawKeypoints(
    img_2_cv,
    keypoints2,
    None
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_1_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image 1 with Keypoints')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_2_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image 2 with Keypoints')
plt.show()

# Match descriptors using BFMatcher

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 10 matches
im_matches = cv2.drawMatches(img_1_cv, keypoints1, img_2_cv, keypoints2, matches[:10], None)

# Show matches
plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(im_matches, cv2.COLOR_BGR2RGB))
plt.title('Top 10 Matches')
plt.show()

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw good matches
im_good_matches = cv2.drawMatches(img_1_cv, keypoints1, img_2_cv, keypoints2, good_matches, None)

# Show good matches
plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(im_good_matches, cv2.COLOR_BGR2RGB))
plt.title('Good Matches after Ratio Test')
plt.show()
