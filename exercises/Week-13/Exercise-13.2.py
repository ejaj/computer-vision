import cv2
import numpy as np
from matplotlib import pyplot as plt

c = np.load('data/casper/calib.npy', allow_pickle=True).item()
im0 = cv2.imread("data/casper/sequence/frames0_0.png")
size = (im0.shape[1], im0.shape[0])
# print(size)

stereo = cv2.stereoRectify(c['K0'], c['d0'], c['K1'],
                           c['d1'], size, c['R'], c['t'], flags=0)
# print(stereo)

R0, R1, P0, P1 = stereo[:4]

maps0 = cv2.initUndistortRectifyMap(c['K0'], c['d0'], R0, P0, size, cv2.CV_32FC2)
maps1 = cv2.initUndistortRectifyMap(c['K1'], c['d1'], R1, P1, size, cv2.CV_32FC2)

# Initialize lists to store rectified images
ims0 = []
ims1 = []

for i in range(26):
    # Load image for camera 0
    im0 = cv2.imread(f'data/casper/sequence/frames0_{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # Load image for camera 1
    im1 = cv2.imread(f'data/casper/sequence/frames1_{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Rectify images
    im0_rectified = cv2.remap(im0, maps0[0], maps0[1], cv2.INTER_LINEAR)
    im1_rectified = cv2.remap(im1, maps1[0], maps1[1], cv2.INTER_LINEAR)

    # Store rectified images
    ims0.append(im0_rectified)
    ims1.append(im1_rectified)

# Visualize the first rectified images from both cameras
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ims0[0], cmap='gray')
plt.title("Camera 0 - Rectified")

plt.subplot(1, 2, 2)
plt.imshow(ims1[0], cmap='gray')
plt.title("Camera 1 - Rectified")

plt.show()
