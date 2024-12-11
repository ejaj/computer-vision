import cv2
import numpy as np
import matplotlib.pyplot as plt
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian

# Load grayscale image
image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/dog.jpg", cv2.IMREAD_GRAYSCALE)

# Create a blank scribbles mask
scribbles = np.zeros_like(image)

# Add scribbles for the foreground (value = 1) and background (value = 2)
scribbles[50:150, 100:200] = 1  # Foreground
scribbles[0:50, 0:100] = 2      # Background

# Map scribbles to binary labels [0, 1] for the CRF
labels = np.zeros_like(scribbles)
labels[scribbles == 1] = 0  # Foreground class
labels[scribbles == 2] = 1  # Background class

# Prepare CRF
crf = DenseCRF2D(image.shape[1], image.shape[0], 2)  # 2 classes: background and foreground
unary = unary_from_labels(labels, 2, gt_prob=0.7)  # Unary potentials from labels
crf.setUnaryEnergy(unary)
pairwise = create_pairwise_gaussian((3, 3), image.shape[:2])  # Smoothness pairwise potentials
crf.addPairwiseEnergy(pairwise, compat=10)

# Perform inference (labeling)
output = crf.inference(5)  # 5 iterations
segmentation = np.argmax(output, axis=0).reshape(image.shape[:2])

# Show results
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Scribbles")
plt.imshow(scribbles, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Segmented Image")
plt.imshow(segmentation, cmap='gray')

plt.show()
