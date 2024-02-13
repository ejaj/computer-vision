import numpy as np


def normalize2d(pts):
    """
    Normalizes a set of points so that the centroid of the points is at the origin
    and the average distance to the origin is sqrt(2).

    Args:
    pts (numpy.ndarray): The set of 2D points to be normalized (size: Nx2).

    Returns:
    numpy.ndarray: The normalized points (size: Nx3 in homogeneous coordinates).
    numpy.ndarray: The transformation matrix used to normalize the points.
    """

    # Calculate the centroid of the points
    centroid = np.mean(pts, axis=0)

    # Translate points to have centroid at the origin
    pts_centered = pts - centroid

    # Calculate the average distance of the points from the origin
    dists = np.sqrt(np.sum(pts_centered ** 2, axis=1))
    mean_dist = np.mean(dists)

    # Calculate the scale factor so that the mean distance is sqrt(2)
    scale = np.sqrt(2) / mean_dist

    # Create the normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    # Normalize the points by applying the transformation matrix
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_normalized = (T @ pts_homogeneous.T).T

    return pts_normalized, T
