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


# Load the image
image_path = 'data/TestIm1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply smoothed Hessian
sigma = 2.0
epsilon = 1.0
C = smoothedHessian(image, sigma, epsilon)

# Display the smoothed Hessian matrix components
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title('C[0, 0]')
plt.imshow(C[:, :, 0, 0], cmap='gray')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title('C[0, 1]')
plt.imshow(C[:, :, 0, 1], cmap='gray')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title('C[1, 0]')
plt.imshow(C[:, :, 1, 0], cmap='gray')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title('C[1, 1]')
plt.imshow(C[:, :, 1, 1], cmap='gray')
plt.colorbar()

plt.show()
