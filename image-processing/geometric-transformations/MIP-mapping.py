import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_mip_map(image, levels):
    """Generate MIP-map levels for a given image."""
    mip_map = [image]
    current = image
    for i in range(1, levels):
        current = cv2.pyrDown(current)  # Downsample image by 2x
        mip_map.append(current)
    return mip_map

def display_mip_map(mip_map):
    """Display MIP-map levels."""
    plt.figure(figsize=(10, 5))
    for i, level in enumerate(mip_map):
        plt.subplot(1, len(mip_map), i + 1)
        plt.imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
        plt.title(f"Level {i}")
        plt.axis("off")
    plt.show()

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/grid.png", cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (512, 512))  # Resize to a standard size

# Create MIP-map with 4 levels
mip_map = create_mip_map(image, levels=4)

# Display the MIP-map levels
display_mip_map(mip_map)
