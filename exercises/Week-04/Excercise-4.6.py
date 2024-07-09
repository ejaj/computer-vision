import numpy as np


def estimate_b(Hs):
    """
    Estimate the vector b from a list of homographies Hs.

    Args:
    Hs (list of np.array): List of 3x3 homography matrices.

    Returns:
    np.array: The vector b derived from SVD.
    """
    V = []

    for H in Hs:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h1h1 = np.dot(h1, h1)  # h1^T * h1
        h2h2 = np.dot(h2, h2)  # h2^T * h2
        h1h2 = np.dot(h1, h2)  # h1^T * h2

        V.append([h1h1, h2h2, 2 * h1h2])
        V.append([h1h1 - h2h2, 0, 0])

    V = np.array(V)
    U, s, Vt = np.linalg.svd(V)
    b = Vt[-1]  # The last row of Vt corresponds to the smallest singular value

    return b


# Example usage:
# Assuming Hs is a list of homography matrices (3x3) derived from previous exercises
Hs = [np.eye(3) for _ in range(3)]  # Placeholder for actual homographies
b = estimate_b(Hs)
print("Estimated vector b:", b)
