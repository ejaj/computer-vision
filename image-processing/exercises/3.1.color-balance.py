import cv2
import numpy as np
from matplotlib import pyplot as plt

# Gamma correction function
def gamma_correction(image, gamma=2.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/naymer.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
print(image.shape)

# User-specified constants for red, green, blue channels
red_multiplier = 1.2
green_multiplier = 1.0
blue_multiplier = 0.8

balanced_image = image.astype(np.float32)  # Convert to float for accurate scaling
balanced_image[:, :, 0] *= blue_multiplier  # Blue channel
balanced_image[:, :, 1] *= green_multiplier  # Green channel
balanced_image[:, :, 2] *= red_multiplier  # Red channel

# Clip values to [0, 255] and convert back to uint8
balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)

# Display images
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Color Balanced Image")
plt.imshow(balanced_image)
plt.show()


# Gamma correction

# Apply gamma correction before multiplication
gamma_corrected_before = gamma_correction(image)

# Apply color balance on gamma-corrected image
balanced_image_before = gamma_corrected_before.astype(np.float32)
balanced_image_before[:, :, 0] *= blue_multiplier
balanced_image_before[:, :, 1] *= green_multiplier
balanced_image_before[:, :, 2] *= red_multiplier
balanced_image_before = np.clip(balanced_image_before, 0, 255).astype(np.uint8)

# Apply color balance first, then gamma correction
balanced_image_after = balanced_image.astype(np.float32)
balanced_image_after[:, :, 0] *= blue_multiplier
balanced_image_after[:, :, 1] *= green_multiplier
balanced_image_after[:, :, 2] *= red_multiplier
balanced_image_after = np.clip(balanced_image_after, 0, 255).astype(np.uint8)
gamma_corrected_after = gamma_correction(balanced_image_after)

# Display results
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.title("Gamma Before Multiplication")
plt.imshow(balanced_image_before)

plt.subplot(1, 3, 3)
plt.title("Gamma After Multiplication")
plt.imshow(gamma_corrected_after)

plt.show()



