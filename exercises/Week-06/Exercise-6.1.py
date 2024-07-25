import numpy as np
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def gaussian1DKernel(sigma):
    length = int(np.ceil(6 * sigma))
    if length % 2 == 0:
        length += 1
    half_length = length // 2
    x = np.arange(-half_length, half_length + 1)
    g = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- (x ** 2) / (2 * sigma ** 2))
    g = g / np.sum(g)  # Normalize to sum to 1
    gd = -x / (sigma ** 2) * g
    return g, gd


# # Test the function
# sigma = 1.0
# g, gd = gaussian1DKernel(sigma)
# print("Gaussian Kernel (g):", g)
# print("Derivative of Gaussian Kernel (gd):", gd)

def compute_image_gradients(image, sigma):
    g, gd = gaussian1DKernel(sigma)

    Ix = convolve(image, gd[:, None])
    Iy = convolve(image, gd[None, :])

    return Ix, Iy


image = cv2.imread('data/TestIm1.png', cv2.IMREAD_GRAYSCALE)

sigma = 1.0
Ix, Iy = compute_image_gradients(image, sigma)
print(Ix, Iy)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Gradient in x direction')
plt.imshow(Ix, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Gradient in y direction')
plt.imshow(Iy, cmap='gray')
plt.show()


def compute_gradient_products(Ix, Iy):
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    return Ix2, Iy2, Ixy


Ix2, Iy2, Ixy = compute_gradient_products(Ix, Iy)


def apply_gaussian_filter(Ix2, Iy2, Ixy, sigma):
    g, _ = gaussian1DKernel(sigma)
    Sx2 = convolve(Ix2, g[:, None])
    Sx2 = convolve(Sx2, g[None, :])

    Sy2 = convolve(Iy2, g[:, None])
    Sy2 = convolve(Sy2, g[None, :])

    Sxy = convolve(Ixy, g[:, None])
    Sxy = convolve(Sxy, g[None, :])

    return Sx2, Sy2, Sxy


Sx2, Sy2, Sxy = apply_gaussian_filter(Ix2, Iy2, Ixy, sigma)


def compute_harris_response(Sx2, Sy2, Sxy, k=0.04):
    detM = Sx2 * Sy2 - Sxy ** 2
    traceM = Sx2 + Sy2
    R = detM - k * (traceM ** 2)
    return R


R = compute_harris_response(Sx2, Sy2, Sxy)


def threshold_and_nms(R, threshold, window_size=3):
    corner_peaks = (R > threshold) * R
    coordinates = np.array(np.nonzero(corner_peaks)).T
    corners = []

    for coord in coordinates:
        x, y = coord
        if R[x, y] == np.max(
                R[x - window_size // 2:x + window_size // 2 + 1, y - window_size // 2:y + window_size // 2 + 1]):
            corners.append((x, y))

    return corners


corners = threshold_and_nms(R, threshold=1e5)

# Display corners on the image
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.scatter([c[1] for c in corners], [c[0] for c in corners], color='red')
plt.title('Corners detected')
plt.show()
