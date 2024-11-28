import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Analyze the frequency content of an image.

The Fourier Transform converts the image to its frequency domain.
The magnitude spectrum shows the frequency content:
Center: Low frequencies (smooth areas).
Edges: High frequencies (edges, fine details).

"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

# Compute the Fourier Transform and shift the zero frequency component to the center
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# Compute the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(f_shift))


# Display the gradients
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Orginal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis("off")

plt.tight_layout()
plt.show()
