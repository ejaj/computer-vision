import cv2
import numpy as np
from matplotlib import pyplot as plt

# foreground = cv2.imread('/home/kazi/Works/Dtu/computer-vision/data/face.png')

# # Create a dummy alpha channel (example: a circular mask)
# alpha = np.zeros((foreground.shape[0], foreground.shape[1]), dtype=np.uint8)
# cv2.circle(alpha, (foreground.shape[1] // 2, foreground.shape[0] // 2), 100, 255, -1)

# # Save alpha mask and composite with the foreground
# foreground_with_alpha = cv2.merge((foreground, alpha))
# cv2.imwrite('/home/kazi/Works/Dtu/computer-vision/data/face_alpha.png', foreground_with_alpha)

# Function for gamma correction
def apply_gamma(image, gamma):
    # Scale image to [0, 255] and convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Create LUT (look-up table) for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    
    # Apply LUT and return the result, rescaling to [0, 1]
    corrected = cv2.LUT(image_uint8, table)
    return corrected.astype(np.float32) / 255

# Load images
foreground = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/face.png").astype(np.float32) / 255
background = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/bg.png").astype(np.float32) / 255
alpha = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/face_alpha.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

# Resize all images to the same dimensions
target_height, target_width = alpha.shape
foreground = cv2.resize(foreground, (target_width, target_height))
background = cv2.resize(background, (target_width, target_height))

# Perform compositing in linear space
foreground_linear = apply_gamma(foreground, gamma=2.2)
background_linear = apply_gamma(background, gamma=2.2)

composited_linear = alpha[:, :, None] * foreground_linear + (1 - alpha[:, :, None]) * background_linear

# Convert back to gamma-corrected space
composited_gamma = np.clip(composited_linear ** (1 / 2.2), 0, 1) * 255

# Display the result
plt.imshow(composited_gamma.astype(np.uint8))
plt.title("Composited Image")
plt.axis('off')
plt.show()


# tow

# Load foreground (with alpha) and background
foreground = cv2.imread('/home/kazi/Works/Dtu/computer-vision/data/face_alpha.png', cv2.IMREAD_UNCHANGED)  # Includes alpha channel
background = cv2.imread('/home/kazi/Works/Dtu/computer-vision/data/bg.png')

# Extract the alpha channel from the foreground
alpha = foreground[:, :, 3] / 255.0  # Normalize alpha to range [0, 1]

# Resize background to match foreground dimensions
background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

# Perform alpha compositing
foreground_rgb = foreground[:, :, :3]
composite = np.uint8(
    alpha[:, :, None] * foreground_rgb + (1 - alpha[:, :, None]) * background
)

# Display results
plt.subplot(1, 2, 1)
plt.title("Foreground")
plt.imshow(cv2.cvtColor(foreground_rgb, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Composited Image")
plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
plt.show()