import numpy as np


def triangulate_n(qs, Ps):
    """
    Triangulate a single 3D point from its projections in n views using a linear algorithm.

    Parameters:
    - qs: List of n pixel coordinates in homogeneous form (x, y, 1).
    - Ps: List of n projection matrices.

    Returns:
    - The 3D coordinates of the triangulated point in homogeneous form.
    """
    # Create an empty matrix for the linear system
    A = np.zeros((len(qs) * 2, 4))

    for i, (q, P) in enumerate(zip(qs, Ps)):
        # For each view, add two equations to the system
        A[2 * i] = q[0] * P[2] - P[0]
        A[2 * i + 1] = q[1] * P[2] - P[1]

    # Solve the system using SVD
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]  # The solution is the last row of Vt

    # Homogenize (make the last component 1)
    return X / X[-1]
