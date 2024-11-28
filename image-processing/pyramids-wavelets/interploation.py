import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
1. Nearest Neighbor Interpolation
Description:

Simply assigns the value of the nearest pixel to the new location.
Produces blocky, pixelated results, especially when scaling up.
When to Use:

Speed is critical: It's the fastest method, requiring minimal computation.
Discrete data: Ideal for images with distinct categories or labels (e.g., masks, segmentation maps, binary images).
Example: Upscaling a binary mask where you don’t want blended (non-binary) pixel values.
Retro aesthetics: When you want to retain a pixelated look, such as in pixel art or retro games.
Why:

Simplicity and speed.
Maintains sharp transitions without introducing new intermediate values, which is crucial for categorical data.
2. Bilinear Interpolation
Description:

Uses linear interpolation between adjacent pixels in both horizontal and vertical directions.
Produces smoother results compared to nearest neighbor.
When to Use:

Real-time graphics: For resizing textures or images in games or user interfaces where speed and smoothness are important.
Basic image processing: When scaling up/down and you need acceptable quality without excessive computation.
Moderate quality requirements: Useful when artifacts like blurriness are acceptable in trade for simplicity.
Why:

Faster than bicubic but produces smoother results than nearest neighbor.
Suitable for applications where sharpness isn't a priority but speed matters.
3. Bicubic Interpolation
Description:

Uses cubic polynomials to interpolate values, considering the nearest 16 pixels (4x4 grid).
Produces smoother, sharper results compared to bilinear interpolation.
When to Use:

High-quality image scaling: Ideal for photo editors, professional graphics tools, and applications requiring visual clarity.
Example: Upscaling photos or videos where sharpness is important.
Image preparation: Before analysis or printing to retain details without introducing blockiness or excessive blur.
Gradual scaling: Best for smoother transitions when scaling up multiple times.
Why:

Balances quality and computational cost.
Produces visually appealing images with sharp edges and smooth gradients.
4. Windowed Sinc Interpolation
Description:

Based on the sinc function, which provides the theoretically best interpolation by minimizing aliasing.
Often combined with a windowing function to reduce oscillations (ringing artifacts).
When to Use:

High-precision applications:
Scientific image processing where detail preservation is critical.
Applications requiring minimal distortion, such as medical imaging or remote sensing.
Upscaling for analysis: When visual and data fidelity is crucial.
Why:

Preserves high-frequency details and avoids aliasing.
Produces the smoothest possible results but can cause visible ringing near sharp edges.

"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/kodim08_grayscale.png", cv2.IMREAD_GRAYSCALE)

# Define upscaling factor
scale = 4
new_size = (image.shape[1] * scale, image.shape[0] * scale)

# 1. Bilinear Interpolation
bilinear_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

# 2. Bicubic Interpolation
bicubic_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

# 3. Nearest Neighbor Interpolation (as a comparison)
nearest_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

# Display the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Bilinear Interpolation")
plt.imshow(bilinear_resized, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Bicubic Interpolation")
plt.imshow(bicubic_resized, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Nearest Neighbor Interpolation")
plt.imshow(nearest_resized, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
