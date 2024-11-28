import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/naymer.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
print(image.shape)

def create_bayer_pattern(image):
    bayer = np.zeros_like(image)
    bayer[::2, ::2, 0] = image[::2, ::2, 0]  # Red channel
    bayer[::2, 1::2, 1] = image[::2, 1::2, 1]  # Green channel (row 1)
    bayer[1::2, ::2, 1] = image[1::2, ::2, 1]  # Green channel (row 2)
    bayer[1::2, 1::2, 2] = image[1::2, 1::2, 2]  # Blue channel
    return bayer

bayer_image = create_bayer_pattern(image)

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Bayer Pattern")
plt.imshow(bayer_image)
plt.show()

def demosaic_bilinear(bayer):
    height, width, _ = bayer.shape
    demosaiced = np.zeros_like(bayer, dtype=np.float32)
    
    # Fill red channel
    demosaiced[:, :, 0] = cv2.resize(bayer[:, :, 0], (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Fill green channel
    demosaiced[:, :, 1] = cv2.resize(bayer[:, :, 1], (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Fill blue channel
    demosaiced[:, :, 2] = cv2.resize(bayer[:, :, 2], (width, height), interpolation=cv2.INTER_LINEAR)
    
    return np.clip(demosaiced, 0, 255).astype(np.uint8)

demosaiced_image = demosaic_bilinear(bayer_image)

plt.subplot(1, 2, 1)
plt.title("Demosaiced Image (Bilinear)")
plt.imshow(demosaiced_image)

plt.subplot(1, 2, 2)
plt.title("Original Image")
plt.imshow(image)
plt.show()