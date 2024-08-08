import cv2
import numpy as np
import matplotlib.pyplot as plt


def estHomographyRANSAC(kp1, des1, kp2, des2, iterations=200, sigma=3):
    # Match features using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Minimum number of matches to estimate a homography
    min_matches = 4

    # Check if there are enough good matches
    if len(good_matches) < min_matches:
        raise ValueError("Not enough matches to compute homography")

    # RANSAC parameters
    threshold = 3 * sigma ** 2

    def ransac_homography(matches, kp1, kp2, iterations, threshold):
        best_homography = None
        max_inliers = 0
        best_inliers = []

        for i in range(iterations):
            # Randomly select 4 matches
            sample_matches = np.random.choice(matches, min_matches, replace=False)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in sample_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in sample_matches])

            # Compute homography
            H, _ = cv2.findHomography(pts1, pts2, 0)

            # Compute inliers
            inliers = []
            for m in matches:
                pt1 = np.float32([kp1[m.queryIdx].pt])
                pt2 = np.float32([kp2[m.trainIdx].pt])
                pt1_transformed = cv2.perspectiveTransform(np.array([pt1]), H)[0][0]
                dist = np.sum((pt2 - pt1_transformed) ** 2)
                if dist < threshold:
                    inliers.append(m)

            # Update the best homography if the current one has more inliers
            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_homography = H
                best_inliers = inliers

        return best_homography, best_inliers

    # Compute the homography and inliers using RANSAC
    H, inliers = ransac_homography(good_matches, kp1, kp2, iterations, threshold)

    # Refit the homography using all inliers
    if len(inliers) > 0:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in inliers])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in inliers])
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    return H, inliers


if __name__ == "__main__":
    # Load the images
    im1 = cv2.imread('data/im1.jpg', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('data/im2.jpg', cv2.IMREAD_GRAYSCALE)

    # Detect SIFT keypoints and compute descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    try:
        # Estimate homography using RANSAC
        H, inliers = estHomographyRANSAC(kp1, des1, kp2, des2)

        # Draw the matches
        matched_image = cv2.drawMatches(im1, kp1, im2, kp2, inliers, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the matched image with inliers
        plt.imshow(matched_image)
        plt.show()
    except ValueError as e:
        print(e)
