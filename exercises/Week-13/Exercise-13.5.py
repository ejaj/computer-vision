import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the calibration data
c = np.load('data/casper/calib.npy', allow_pickle=True).item()
im0 = cv2.imread("data/casper/sequence/frames0_0.png")
size = (im0.shape[1], im0.shape[0])

# Stereo rectification
stereo = cv2.stereoRectify(c['K0'], c['d0'], c['K1'], c['d1'], size, c['R'], c['t'], flags=0)
R0, R1, P0, P1 = stereo[:4]

maps0 = cv2.initUndistortRectifyMap(c['K0'], c['d0'], R0, P0, size, cv2.CV_32FC1)
maps1 = cv2.initUndistortRectifyMap(c['K1'], c['d1'], R1, P1, size, cv2.CV_32FC2)

# Initialize lists to store rectified images
ims0 = []
ims1 = []

# Load and rectify images
for i in range(26):
    im0 = cv2.imread(f'data/casper/sequence/frames0_{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    im1 = cv2.imread(f'data/casper/sequence/frames1_{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    im0_rectified = cv2.remap(im0, maps0[0], maps0[1], cv2.INTER_LINEAR)
    im1_rectified = cv2.remap(im1, maps1[0], maps1[1], cv2.INTER_LINEAR)

    ims0.append(im0_rectified)
    ims1.append(im1_rectified)


def unwrap(ims):
    primary_images = ims[2:18]
    fft_primary = np.fft.rfft(primary_images, axis=0)
    theta_primary = np.angle(fft_primary[1])

    secondary_images = ims[18:26]
    fft_secondary = np.fft.rfft(secondary_images, axis=0)
    theta_secondary = np.angle(fft_secondary[1])

    theta_c = (theta_primary - theta_secondary) % (2 * np.pi)
    o_primary = np.round((theta_primary - theta_secondary) / (2 * np.pi))
    theta = theta_primary + o_primary * 2 * np.pi

    return theta


theta0 = unwrap(ims0)
theta1 = unwrap(ims1)

# Corrected mask calculation
diff0 = ims0[0] - ims0[1]  # For Camera 0
diff1 = ims1[0] - ims1[1]  # For Camera 1

threshold = 15
mask0 = (diff0 > threshold).astype(np.uint8)  # Binary mask for Camera 0
mask1 = (diff1 > threshold).astype(np.uint8)  # Binary mask for Camera 1

# Matching phase values
q0s = []
q1s = []

disparity = np.zeros_like(theta0)

for i0 in range(theta0.shape[0]):  # Iterate over rows
    for j0 in range(theta0.shape[1]):  # Iterate over columns
        if mask0[i0, j0]:  # Check if the pixel is valid in Camera 0
            best_match = None
            min_phase_diff = np.inf  # Initialize with a large value

            # Search on the same row i0 in Camera 1
            for j1 in range(theta1.shape[1]):
                if mask1[i0, j1]:  # Check if the pixel is valid in Camera 1
                    phase_diff = np.abs(theta0[i0, j0] - theta1[i0, j1])

                    if phase_diff < min_phase_diff:
                        min_phase_diff = phase_diff
                        best_match = j1

            if best_match is not None:
                q0s.append([j0, i0])  # [x, y] for Camera 0
                q1s.append([best_match, i0])  # [x, y] for Camera 1

                disparity[i0, j0] = j0 - best_match

# Apply median filtering
disparity_filtered = cv2.medianBlur(disparity.astype(np.float32), 5)

# Visualize the disparity map
plt.figure(figsize=(10, 5))
plt.imshow(disparity_filtered, cmap='viridis')
plt.title('Disparity Map')
plt.colorbar()
plt.show()
