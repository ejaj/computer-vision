import cv2

# Load the images
image1 = cv2.imread('data/im1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('data/im2.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect SIFT keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

# Initialize the Brute-Force matcher with default parameters
bf = cv2.BFMatcher()

# Perform cross-checking
# Match descriptors from image1 to image2 and from image2 to image1
matches1 = bf.knnMatch(des1, des2, k=2)
matches2 = bf.knnMatch(des2, des1, k=2)

# Apply ratio test as per Lowe's paper to filter good matches
good_matches1 = []
for m, n in matches1:
    if m.distance < 0.75 * n.distance:
        good_matches1.append(m)

good_matches2 = []
for m, n in matches2:
    if m.distance < 0.75 * n.distance:
        good_matches2.append(m)
# Cross-check to keep only mutually best matches
cross_checked_matches = []
for match1 in good_matches1:
    for match2 in good_matches2:
        if match1.queryIdx == match2.trainIdx and match1.trainIdx == match2.queryIdx:
            cross_checked_matches.append(match1)
            break

# Draw the matches
matched_image = cv2.drawMatches(image1, kp1, image2, kp2, cross_checked_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched image
cv2.imshow('Matched keypoints', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
