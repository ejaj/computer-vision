import numpy as np


def normalize2d(pts):
    """
    Normalize a set of points so that the centroid of the points is at the origin and
    the average distance to the origin is sqrt(2).
    """
    mean = np.mean(pts, axis=0)
    std_dev = np.std(pts)
    scale = np.sqrt(2) / std_dev

    # Transformation matrix
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])

    # Apply transformation
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_t = (T @ pts_h.T).T

    return pts_t, T


def hest(q1, q2, normalize=True):
    """
    Estimate the homography matrix using the linear algorithm.
    If normalize is True, the points will be normalized before the estimation.
    """
    if normalize:
        q1, T1 = normalize2d(q1)
        q2, T2 = normalize2d(q2)
    else:
        # Create homogenous coordinates if not normalizing
        q1 = np.hstack((q1, np.ones((q1.shape[0], 1))))
        q2 = np.hstack((q2, np.ones((q2.shape[0], 1))))

    # Construct the A matrix
    A = []
    for i in range(q1.shape[0]):
        x1, y1, w1 = q1[i]
        x2, y2, w2 = q2[i]
        A.append([-x1, -y1, -w1, 0, 0, 0, x2 * x1, x2 * y1, x2 * w1])
        A.append([0, 0, 0, -x1, -y1, -w1, y2 * x1, y2 * y1, y2 * w1])
    A = np.array(A)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    # The homography is the last column of V (or the last row of V transposed)
    H = Vt[-1].reshape(3, 3)

    if normalize:
        # Denormalize the homography matrix
        H = np.linalg.inv(T2) @ H @ T1

    return H

# Example usage:
# q1 = np.array([[x1, y1], [x2, y2], ...])
# q2 = np.array([[x'1, y'1], [x'2, y'2], ...])
# H = hest(q1, q2, normalize=True)
