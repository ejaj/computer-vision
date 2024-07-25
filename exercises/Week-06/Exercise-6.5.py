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


def gaussianSmoothing(im, sigma):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)

    if sigma == 0:
        I = im
        Ix = np.zeros_like(im)
        Iy = np.zeros_like(im)
        return I, Ix, Iy

    g, gd = gaussian1DKernel(sigma)

    # Apply Gaussian smoothing
    I = convolve(im, g[:, None])
    I = convolve(I, g[None, :])

    # Compute gradients
    Ix = convolve(im, gd[:, None])
    Ix = convolve(Ix, g[None, :])

    Iy = convolve(im, gd[None, :])
    Iy = convolve(Iy, g[:, None])

    return I, Ix, Iy


def smoothedHessian(im, sigma, epsilon):
    # Compute the gradients using Gaussian smoothing with sigma
    _, Ix, Iy = gaussianSmoothing(im, sigma)

    # Compute products of gradients
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy

    # Apply Gaussian smoothing with epsilon
    g_epsilon, _ = gaussian1DKernel(epsilon)
    Sx2 = convolve(Ix2, g_epsilon[:, None])
    Sx2 = convolve(Sx2, g_epsilon[None, :])

    Sy2 = convolve(Iy2, g_epsilon[:, None])
    Sy2 = convolve(Sy2, g_epsilon[None, :])

    SIxIy = convolve(IxIy, g_epsilon[:, None])
    SIxIy = convolve(SIxIy, g_epsilon[None, :])

    # Form the Hessian matrix C
    C = np.zeros((im.shape[0], im.shape[1], 2, 2))
    C[:, :, 0, 0] = Sx2
    C[:, :, 0, 1] = SIxIy
    C[:, :, 1, 0] = SIxIy
    C[:, :, 1, 1] = Sy2

    return C


def harrisMeasure(im, sigma, epsilon, k=0.06):
    # Compute the smoothed Hessian matrix
    C = smoothedHessian(im, sigma, epsilon)

    # Extract elements of the Hessian matrix
    a = C[:, :, 0, 0]
    b = C[:, :, 1, 1]
    c = C[:, :, 0, 1]

    # Compute the Harris response
    detC = a * b - c * c
    traceC = a + b
    r = detC - k * (traceC ** 2)

    return r


def non_maximum_suppression(r):
    # Create a zero array of the same shape as the Harris response
    r_max = np.zeros_like(r)

    # Compare each point to its neighbors
    for y in range(1, r.shape[0] - 1):
        for x in range(1, r.shape[1] - 1):
            if r[y, x] > r[y - 1, x] and r[y, x] > r[y + 1, x] and r[y, x] > r[y, x - 1] and r[y, x] > r[y, x + 1]:
                r_max[y, x] = r[y, x]

    return r_max


def cornerDetector(im, sigma, epsilon, k, tau):
    # Compute the Harris response
    r = harrisMeasure(im, sigma, epsilon, k)

    # Apply non-maximum suppression
    r_nms = non_maximum_suppression(r)

    # Apply thresholding
    r_nms[r_nms <= tau] = 0

    # Get coordinates of the corners
    corners = np.where(r_nms > 0)
    corner_points = list(zip(corners[0], corners[1]))

    return corner_points


# Load the image
image_path = 'data/TestIm1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Detect corners
sigma = 2.0
epsilon = 1.0
k = 0.06
tau = 1e-6
corners = cornerDetector(image, sigma, epsilon, k, tau)

# Display the image with detected corners
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
for y, x in corners:
    plt.scatter(x, y, c='red', s=10)
plt.title('Detected Corners')
plt.show()
