from scipy.spatial.transform import Rotation

# Define the intrinsic matrix K.
K = [[1000, 0, 300],
     [0, 1000, 200],
     [0, 0, 1]]

# Define the rotation for Cam2 using the given angles.
angles = [0.7, -0.5, 0.8]  # angles in radians for rotation 'xyz'
R2 = Rotation.from_euler('xyz', angles).as_matrix()

# Define the translation for Cam2.
t2 = [0.2, 0, 1]

print("R2", R2)
print("t2", t2)
