import cv2
import numpy as np

# Load the image
image_path = 'data/gopro_robot.jpg'
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

focal_length_pixels = 0.455732 * image_width
c_x = image_width / 2
c_y = image_height / 2

s = 0

K = [
    [focal_length_pixels, s, c_x],
    [0, focal_length_pixels, c_y],
    [0, 0, 1]
]

print("Intrinsic matrix K:")
print(K)
# dist_coeffs = np.array([0, 0, 0, 0, -0.245031, 0, 0, 0.071524, -0.00994978])
