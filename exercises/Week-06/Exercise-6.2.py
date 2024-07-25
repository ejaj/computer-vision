import numpy as np
import cv2
from scipy.ndimage import convolve


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


# Load the image
image_path = 'data/TestIm1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian smoothing
sigma = 5
I, Ix, Iy = gaussianSmoothing(image, sigma)

# Display the results
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title('Smoothed Image (I)')
plt.imshow(I, cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title('Gradient in x direction (Ix)')
plt.imshow(Ix, cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title('Gradient in y direction (Iy)')
plt.imshow(Iy, cmap='gray')
plt.colorbar()

plt.show()