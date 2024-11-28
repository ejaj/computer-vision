import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Use Case: Identifying and labeling distinct regions in a binary image.
Purpose: Separate regions for further analysis.
"""

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/ae.png", cv2.IMREAD_GRAYSCALE)

num_labels, labels = cv2.connectedComponents(image)
print(num_labels, labels)
# Display results
print(f"Number of Connected Components (including background): {num_labels}")
plt.imshow(labels, cmap='jet') 
plt.title("Connected Components")
plt.axis('off')
plt.show()
