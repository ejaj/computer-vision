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


def estimate_b(Hs):
    """ Estimate the vector b from a list of homographies Hs. """
    V = []
    for H in Hs:
        h1, h2 = H[:, 0], H[:, 1]
        v12 = np.dot(h1, h2)
        v11 = np.dot(h1, h1)
        v22 = np.dot(h2, h2)
        V.append([v11, v22, 2 * v12])
        V.append([v11 - v22, 0, 0])
    V = np.array(V)
    U, s, Vt = np.linalg.svd(V)
    b = Vt[-1] / Vt[-1][-1]
    return b


def estimate_intrinsics(Hs):
    """ Estimate the intrinsic camera matrix K from a list of homography matrices. """
    b = estimate_b(Hs)  # Get the vector b from the homographies

    # Fill in the elements of matrix B
    B = np.array([
        [b[0], b[1], b[2]],
        [b[1], b[3], b[4]],
        [b[2], b[4], b[5]]
    ])

    # Calculate the inverse of B
    B_inv = np.linalg.inv(B)

    # Cholesky decomposition of B_inv
    try:
        K_inv = np.linalg.cholesky(B_inv)
        K = np.linalg.inv(K_inv.T)  # Invert the upper triangular matrix from Cholesky
        K = K / K[2, 2]  # Normalize so that the bottom-right element is 1
        return K
    except np.linalg.LinAlgError:
        print("Matrix B_inv is not positive definite and cannot be Choleskied.")
        return None


def estimateExtrinsics(K, Hs):
    """
    Estimate the extrinsic parameters (rotation matrices and translation vectors)
    from the intrinsic camera matrix and a list of homography matrices.

    Args:
    K (np.array): The 3x3 intrinsic camera matrix.
    Hs (list of np.array): List of 3x3 homography matrices.

    Returns:
    tuple: A tuple containing a list of rotation matrices and a list of translation vectors.
    """
    K_inv = np.linalg.inv(K)
    Rs = []
    ts = []

    for H in Hs:
        h1, h2, h3 = K_inv @ H[:, 0], K_inv @ H[:, 1], K_inv @ H[:, 2]
        lamb = 1 / np.linalg.norm(h1)
        r1 = lamb * h1
        r2 = lamb * h2
        r3 = np.cross(r1, r2)
        t = lamb * h3
        R = np.column_stack((r1, r2, r3))
        Rs.append(R)
        ts.append(t)

    return Rs, ts


def calibrateCamera(qs, Q):
    """
    Perform complete camera calibration to find the camera intrinsics and extrinsics.

    Args:
    qs (list of np.array): Projected points in image coordinates for different views.
    Q (np.array): 3D checkerboard points in world coordinates.

    Returns:
    tuple: Camera matrix, list of rotation matrices, and list of translation vectors.
    """
    # First, estimate homographies
    Hs = estimate_homographies(Q, qs)  # assuming estimateHomographies is defined

    # Estimate the intrinsic camera matrix
    K = estimate_intrinsics(Hs)  # assuming estimate_intrinsics is defined

    # Estimate the extrinsic parameters
    Rs, ts = estimateExtrinsics(K, Hs)

    return K, Rs, ts


# Example usage:
K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # Example intrinsic matrix
Hs = [np.eye(3) for _ in range(3)]  # Example homographies; replace with actual homographies

Rs, ts = estimateExtrinsics(K, Hs)
print("Rotations:", Rs)
print("Translations:", ts)
