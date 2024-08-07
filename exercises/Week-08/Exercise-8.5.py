import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def transformIm(im, theta, s):
    width, height = im.size
    new_width = int(width * s)
    new_height = int(height * s)
    scaled_im = im.resize((new_width, new_height), Image.LANCZOS)
    r_im = scaled_im.rotate(theta, expand=True)
    return r_im


im = Image.open('data/sunflowers.jpg')

theta = 45
s = 0.5
transformed_im = transformIm(im, theta, s)

im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

transformed_im_cv = cv2.cvtColor(np.array(transformed_im), cv2.COLOR_RGB2BGR)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(im_cv, None)
keypoints2, descriptors2 = sift.detectAndCompute(transformed_im_cv, None)

im_with_keypoints = cv2.drawKeypoints(im_cv, keypoints1, None)
transformed_im_with_keypoints = cv2.drawKeypoints(transformed_im_cv, keypoints2, None)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Original Image with Keypoints')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(transformed_im_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Transformed Image with Keypoints')
plt.show()

# Match descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 10 matches
im_matches = cv2.drawMatches(im_cv, keypoints1, transformed_im_cv, keypoints2, matches[:10], None)

# Show matches
plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(im_matches, cv2.COLOR_BGR2RGB))
plt.title('Top 10 Matches')
plt.show()

# Apply ratio test as per Lowe's paper
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw good matches
im_good_matches = cv2.drawMatches(im_cv, keypoints1, transformed_im_cv, keypoints2, good_matches, None)

# Show good matches
plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(im_good_matches, cv2.COLOR_BGR2RGB))
plt.title('Good Matches after Ratio Test')
plt.show()
