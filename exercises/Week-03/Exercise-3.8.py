import numpy as np

data = np.load('data/TwoImageData.npy', allow_pickle=True).item()


# Compute the cross product matrix for the translation vector
def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


R1 = data['R1']
R2 = data['R2']
t1 = data['t1']
t2 = data['t2']
K = data['K']

# Compute relative rotation and translation from camera 1 to camera 2
R = R2 @ R1.T  # Rotation from camera 1 to camera 2
t = np.array(t2 - t1).reshape(-1)  # Translation from camera 1 to camera 2

# Compute the essential matrix E
E = skew_symmetric(t) @ R
F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
print(F)
