import numpy as np
import cv2
from scipy.ndimage import convolve
from matplotlib import pyplot as plt


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


# Load the image
image_path = 'data/TestIm1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply the Harris measure
sigma = 2.0
epsilon = 1.0
k = 0.06
r = harrisMeasure(image, sigma, epsilon, k)

# Display the Harris response
plt.figure(figsize=(8, 8))
plt.title('Harris Response (r)')
plt.imshow(r, cmap='jet')
plt.colorbar()
plt.show()
