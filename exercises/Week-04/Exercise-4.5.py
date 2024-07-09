import numpy as np


def estimate_homographies(Q_omega, qs):
    """
    Estimate homography matrices that map 3D checkerboard points (flattened to 2D) to various 2D projections.

    Args:
    Q_omega (np.array): 3xN array of 3D checkerboard points.
    qs (list of np.array): List of 2xN arrays, each containing 2D projections of Q_omega for different views.

    Returns:
    list of np.array: List of 3x3 homography matrices.
    """
    # Convert 3D points Q_omega to 2D points in homogeneous coordinates
    Q_tilde_omega = np.vstack([Q_omega[0, :], Q_omega[1, :], np.ones(Q_omega.shape[1])])

    Hs = []  # List to store homography matrices

    for q in qs:
        # Create matrix A for the DLT solution
        A = []
        for i in range(Q_tilde_omega.shape[1]):
            X, Y, W = Q_tilde_omega[:, i]
            x, y = q[:, i]
            A.append([-X, -Y, -W, 0, 0, 0, x * X, x * Y, x * W])
            A.append([0, 0, 0, -X, -Y, -W, y * X, y * Y, y * W])

        A = np.array(A)

        # Solve for homography using SVD
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape((3, 3))

        # Normalize the homography matrix
        H /= H[2, 2]

        Hs.append(H)

    return Hs


# Example Usage
# Assuming Q_omega is a 3xN array where N is the number of points
# qs is a list of 2xN arrays corresponding to the projections of Q_omega
Q_omega = np.random.rand(3, 10)  # Random 3D points for example
qs = [np.random.rand(2, 10) for _ in range(3)]  # Simulated different 2D projections for example

homographies = estimate_homographies(Q_omega, qs)
for i, H in enumerate(homographies):
    print(f"Homography {i + 1}:\n{H}\n")
