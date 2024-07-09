import numpy as np


def pest(Q, q):
    # Number of points
    n = Q.shape[1]

    # Create matrix A for the DLT solution
    A = []
    for i in range(n):
        X, Y, Z = Q[:, i]
        x, y = q[:, i]
        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y])

    A = np.array(A)

    # Perform SVD on A
    U, S, Vt = np.linalg.svd(A)
    V = Vt.T

    # Last column of V gives us the solution to A @ p = 0
    p_est = V[:, -1]
    P_est = p_est.reshape(3, 4)  # Reshape into a 3x4 matrix

    # Reproject points using P_est
    q_est = P_est @ np.vstack((Q, np.ones((1, n))))
    q_est /= q_est[2, :]  # Normalize

    # Calculate reprojection error (RMSE)
    errors = np.linalg.norm(q_est[:2, :] - q, axis=0)
    rmse = np.sqrt(np.mean(errors ** 2))

    return P_est, q_est, rmse


# Example usage:
# 3D points (in homogeneous coordinates)
Q = np.array([
    [0, 0, 1, 1, 2],  # X coordinates
    [0, 1, 1, 2, 2],  # Y coordinates
    [1, 1, 0, 1, 2]  # Z coordinates
])

# Corresponding 2D projections
q = np.array([
    [1, 2, 3, 4, 5],  # x coordinates
    [1, 2, 2, 3, 3]  # y coordinates
])

P_est, q_est, rmse = pest(Q, q)
print("Estimated P:", P_est)
print("Reprojected q:", q_est)
print("Reprojection Error (RMSE):", rmse)
