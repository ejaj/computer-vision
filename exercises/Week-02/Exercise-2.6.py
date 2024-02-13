import numpy as np


def hest(q1, q2):
    # Ensure that points are in homogeneous coordinates
    q1_h = np.hstack((q1, np.ones((q1.shape[0], 1))))
    q2_h = np.hstack((q2, np.ones((q2.shape[0], 1))))

    # Construct the matrix A for the linear system
    A = []
    for i in range(q1.shape[0]):
        x, y, w = q1_h[i, :]
        xp, yp, wp = q2_h[i, :]
        A.append([-x, -y, -w, 0, 0, 0, x * xp, y * xp, w * xp])
        A.append([0, 0, 0, -x, -y, -w, x * yp, y * yp, w * yp])
    A = np.array(A)

    # Construct the vector b for the linear system
    b = q2_h.reshape(-1, 1)

    # Solve the linear system A * h = b for h
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Reshape h into the 3x3 homography matrix
    H = h.reshape(3, 3)

    return H
