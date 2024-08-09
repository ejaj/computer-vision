import cv2
import numpy as np
import open3d as o3d

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
colors = []

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

                # Sample color from the original image at the matched point
                color = im0[i0, j0] / 255.0  # Normalize to [0, 1] for Open3D
                colors.append([color, color, color])  # Grayscale to RGB

# Convert to NumPy arrays
q0s_np = np.array(q0s).T.astype(np.float32)
q1s_np = np.array(q1s).T.astype(np.float32)
colors_np = np.array(colors).astype(np.float32)

# Triangulate points using the projection matrices
points_4d_homogeneous = cv2.triangulatePoints(P0, P1, q0s_np, q1s_np)

# Convert to 3D Euclidean coordinates
points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]

# Filter points with positive z-coordinates (in front of the cameras)
valid_indices = points_3d[2] > 0
points_3d = points_3d[:, valid_indices]
colors_np = colors_np[valid_indices]  # Filter colors accordingly

# Transpose for visualization
points_3d = points_3d.T

# Visualize with color using Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors_np)
o3d.visualization.draw_geometries([pcd])
