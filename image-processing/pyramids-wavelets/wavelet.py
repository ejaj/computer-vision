import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

"""
When to Use Wavelets
Denoising: Clean signals/images while preserving edges (e.g., medical imaging).
Compression: Reduce data size efficiently (e.g., JPEG2000).
Feature Detection: Identify edges, textures, or patterns (e.g., object recognition).
Time-Frequency Analysis: Analyze changing signals (e.g., ECG, seismic data).
Multi-Resolution Tasks: Study data at global and local scales (e.g., satellite images).
Blending: Seamlessly merge images (e.g., panoramas).
Why Use Wavelets
Localization: Capture spatial and frequency details.
Multi-Resolution: Analyze coarse and fine details simultaneously.
Noise Reduction: Remove noise without blurring edges.
Efficient Representation: Compress data with fewer coefficients.
Reconstruction: Enable perfect data recovery.

"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/lena.png", cv2.IMREAD_GRAYSCALE)

# Perform a single-level 2D Discrete Wavelet Transform
wavelet = 'haar'  # Choose wavelet type ('haar', 'db1', etc.)
coeffs = pywt.dwt2(image, wavelet)  # Decompose the image

# Extract approximation and detail coefficients
cA, (cH, cV, cD) = coeffs

# Display the components
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(2, 3, 2)
plt.title("Approximation (Low-Low)")
plt.imshow(cA, cmap='gray')

plt.subplot(2, 3, 3)
plt.title("Horizontal Detail (High-Low)")
plt.imshow(cH, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("Vertical Detail (Low-High)")
plt.imshow(cV, cmap='gray')

plt.subplot(2, 3, 5)
plt.title("Diagonal Detail (High-High)")
plt.imshow(cD, cmap='gray')

plt.tight_layout()
plt.show()

# Reconstruct the original image from the coefficients
reconstructed_image = pywt.idwt2(coeffs, wavelet)

# Display the reconstructed image
plt.figure(figsize=(6, 6))
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image, cmap='gray')
plt.show()
