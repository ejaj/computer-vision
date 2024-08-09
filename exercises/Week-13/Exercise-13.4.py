import cv2
import numpy as np
from matplotlib import pyplot as plt

c = np.load('data/casper/calib.npy', allow_pickle=True).item()
im0 = cv2.imread("data/casper/sequence/frames0_0.png")
size = (im0.shape[1], im0.shape[0])

stereo = cv2.stereoRectify(
    c['K0'], c['d0'],
    c['K1'], c['d1'],
    size, c['R'], c['t'], flags=0
)

R0, R1, P0, P1 = stereo[:4]
maps0 = cv2.initUndistortRectifyMap(
    c['K0'], c['d0'], R0, P0,
    size, cv2.CV_32FC1
)
maps1 = cv2.initUndistortRectifyMap(
    c['K1'], c['d1'], R1, P1,
    size, cv2.CV_32FC2
)
# Initialize lists to store rectified images
ims0 = []
ims1 = []

for i in range(26):
    im0 = cv2.imread(f'data/casper/sequence/frames0_{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    im1 = cv2.imread(f'data/casper/sequence/frames1_{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Rectify images
    im0_rectified = cv2.remap(im0, maps0[0], maps0[1], cv2.INTER_LINEAR)
    im1_rectified = cv2.remap(im1, maps1[0], maps1[1], cv2.INTER_LINEAR)

    ims0.append(im0_rectified)
    ims1.append(im1_rectified)


def unwrap(ims):
    # Step 1: Extract primary images (images 2 to 17)
    primary_images = ims[2:18]  # Extract 16 images

    # Step 2: Compute the FFT of the primary images
    fft_primary = np.fft.rfft(primary_images, axis=0)

    # Step 3: Get the primary phase
    theta_primary = np.angle(fft_primary[1])

    # Step 4: Extract secondary images (images 18 to 25)
    secondary_images = ims[18:26]  # Extract 8 images

    # Step 5: Compute the FFT of the secondary images
    fft_secondary = np.fft.rfft(secondary_images, axis=0)

    # Step 6: Get the secondary phase
    theta_secondary = np.angle(fft_secondary[1])

    # Step 7: Compute the phase cue using the heterodyne principle
    theta_c = (theta_primary - theta_secondary) % (2 * np.pi)

    # Step 8: Find the order of the primary phase
    o_primary = np.round((theta_primary - theta_secondary) / (2 * np.pi))

    # Step 9: Unwrap the phase
    theta = theta_primary + o_primary * 2 * np.pi

    return theta


theta0 = unwrap(ims0)
theta1 = unwrap(ims1)

diff0 = ims0[0] - ims1[0]
diff1 = ims1[0] - ims0[0]

threshold = 15
mask0 = (diff0 > threshold).astype(np.uint8)  # Binary mask for Camera 0
mask1 = (diff1 > threshold).astype(np.uint8)  # Binary mask for Camera 1

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(mask0, cmap='gray')
plt.title('Mask0 - Camera 0')

plt.subplot(1, 2, 2)
plt.imshow(mask1, cmap='gray')
plt.title('Mask1 - Camera 1')

plt.show()

# Apply masks to the unwrapped phase images
theta0_masked = theta0 * mask0
theta1_masked = theta1 * mask1

# Visualize the masked phase images
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
img0 = plt.imshow(theta0_masked)
plt.title('Theta0 - Masked Unwrapped Phase')
plt.colorbar(img0)

plt.subplot(1, 2, 2)
img1 = plt.imshow(theta1_masked)
plt.title('Theta1 - Masked Unwrapped Phase')
plt.colorbar(img1)
plt.show()
