import cv2
import numpy as np


def find_homography(img1, img2):
    # Find keypoints and descriptors using ORB
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found in one of the images.")
        return None

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        print("Not enough matches found between the images.")
        return None

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Find homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def warp_image(img, H, shape):
    return cv2.warpPerspective(img, H, (shape[1], shape[0]))


# Load images
image1 = cv2.imread('data/1.png')
image2 = cv2.imread('data/2.png')
image3 = cv2.imread('data/3.png')

if image1 is None or image2 is None or image3 is None:
    print("One or more images could not be loaded.")
else:
    # Find pairwise homographies
    H_1_to_2 = find_homography(image1, image2)
    H_2_to_3 = find_homography(image2, image3)

    if H_1_to_2 is None or H_2_to_3 is None:
        print("Homography could not be computed for one or more image pairs.")
    else:
        # Compute composite homography
        H_1_to_3 = np.dot(H_1_to_2, H_2_to_3)

        # Warp images to the reference frame of image2
        warp1_to_2 = warp_image(image1, H_1_to_2, image2.shape)
        warp3_to_2 = warp_image(image3, np.linalg.inv(H_2_to_3), image2.shape)

        # Composite the images (using simple average for demonstration)
        composite_image = np.mean([warp1_to_2, image2, warp3_to_2], axis=0).astype(np.uint8)

        # Save the result
        cv2.imwrite('data/composite_image.jpg', composite_image)
        print("Composite image saved as 'composite_image.jpg'")
