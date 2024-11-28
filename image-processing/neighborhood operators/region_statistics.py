import cv2
import numpy as np
"""
Use Case: Measuring properties of connected regions, such as area, centroid, or moments.	
Purpose: Quantify region characteristics.
"""

# Load labeled image
image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)


# Compute region statistics
unique_labels = np.unique(image)
for label in unique_labels[1:]: 
    component = (image == label).astype(np.uint8)
    area = cv2.countNonZero(component)  
    moments = cv2.moments(component)  
    cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0  
    cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0  

    print(f"Component {label}:")
    print(f" - Area: {area}")
    print(f" - Centroid: ({cx}, {cy})")
