import numpy as np


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


# Example usage:
Hs = [np.eye(3) for _ in range(3)]  # Example data with identity matrices for simplicity
K_estimated = estimate_intrinsics(Hs)
print("Estimated Camera Matrix K:")
print(K_estimated)
