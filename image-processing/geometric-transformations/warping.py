import cv2
import numpy as np

def forward_warping(image, angle):
    # Create an empty destination image
    h, w = image.shape
    dest = np.zeros_like(image)
    angle_rad = np.radians(angle)
    
    # Loop through source image pixels
    for y in range(h):
        for x in range(w):
            # Compute new coordinates using forward rotation
            x_new = int(x * np.cos(angle_rad) - y * np.sin(angle_rad))
            y_new = int(x * np.sin(angle_rad) + y * np.cos(angle_rad))
            
            # Check if the new coordinates are within bounds
            if 0 <= x_new < w and 0 <= y_new < h:
                dest[y_new, x_new] = image[y, x]
    
    return dest

def inverse_warping(image, angle):
    # Create an empty destination image
    h, w = image.shape
    dest = np.zeros_like(image)
    angle_rad = np.radians(angle)
    
    # Loop through destination image pixels
    for y_new in range(h):
        for x_new in range(w):
            # Compute source coordinates using inverse rotation
            x = x_new * np.cos(-angle_rad) - y_new * np.sin(-angle_rad)
            y = x_new * np.sin(-angle_rad) + y_new * np.cos(-angle_rad)
            
            # Use nearest-neighbor interpolation
            x = int(round(x))
            y = int(round(y))
            
            # Check if the source coordinates are within bounds
            if 0 <= x < w and 0 <= y < h:
                dest[y_new, x_new] = image[y, x]
    
    return dest


image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)


# Perform forward and inverse warping
forward_result = forward_warping(image, 45)
inverse_result = inverse_warping(image, 45)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Forward Warping", forward_result)
cv2.imshow("Inverse Warping", inverse_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
