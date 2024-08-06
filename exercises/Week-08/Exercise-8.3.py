import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to compute DoG images
def differenceOfGaussians(im, sigma, n):
    DoG = []
    for i in range(n):
        sigma1 = sigma * (2 ** (i / n))
        sigma2 = sigma * (2 ** ((i + 1) / n))

        g1 = cv2.GaussianBlur(im, (0, 0), sigma1)
        g2 = cv2.GaussianBlur(im, (0, 0), sigma2)

        DoG.append(g1 - g2)

    return DoG


# Function to detect blobs
def detectBlobs(im, sigma, n, tau):
    DoG = differenceOfGaussians(im, sigma, n)
    blobs = []

    # Max filter for non-maximum suppression
    maxDoG = [cv2.dilate(np.abs(DoG[i]), np.ones((3, 3))) for i in range(n)]

    for i in range(n):
        # Thresholding
        mask = (DoG[i] == maxDoG[i]) & (DoG[i] > tau)

        # Check scales above and below
        if i > 0:
            mask &= (DoG[i] >= cv2.dilate(np.abs(DoG[i - 1]), np.ones((3, 3))))
        if i < n - 1:
            mask &= (DoG[i] >= cv2.dilate(np.abs(DoG[i + 1]), np.ones((3, 3))))

        y, x = np.where(mask)
        for j in range(len(x)):
            blobs.append((x[j], y[j], sigma * (2 ** (i / n))))

    return blobs


# Function to draw blobs on the image
def drawBlobs(image, blobs):
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for blob in blobs:
        x, y, s = blob
        radius = int(s * np.sqrt(2))
        cv2.circle(output, (x, y), radius, (0, 0, 255), 2)  # Red color for visibility
    return output


# Load the uploaded image
image_path = 'data/sunflowers.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found")

# Normalize the grayscale image
im = image.astype(float) / 255

# Parameters
sigma = 2
n = 7
tau = 0.1

# Detect blobs
blobs = detectBlobs(im, sigma, n, tau)

# Draw blobs on the image
blob_image = drawBlobs(image, blobs)

# Display the results
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))  # Display in color for visibility
plt.title('Detected Blobs')
plt.axis('off')
plt.show()
