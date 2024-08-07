import numpy as np


def Fest_8point(points1, points2):
    # Number of points
    n = points1.shape[0]

    # Construct matrix A
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = points1[0, i], points1[1, i]
        x2, y2 = points2[0, i], points2[1, i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Compute the SVD of A
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    return F


data = np.load('data/Fest_test.npy', allow_pickle=True).item()
# print(data)
p1 = data['q1']
p2 = data['q2']
Ftrue = data['Ftrue']

# Estimate the fundamental matrix using the Fest_8point function
Fest = Fest_8point(p1, p2)

# Normalize the matrices for comparison
Fest = Fest / np.linalg.norm(Fest)
Ftrue = Ftrue / np.linalg.norm(Ftrue)

# Check if the estimated fundamental matrix is identical to the true matrix up to scale and numerical error
print("Estimated Fundamental Matrix (Fest):")
print(Fest)
print("\nTrue Fundamental Matrix (Ftrue):")
print(Ftrue)
print("\nDifference between Fest and Ftrue (up to scale):")
print(np.abs(Fest - Ftrue))
