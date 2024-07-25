import cv2
import matplotlib.pyplot as plt


def apply_canny(image_path, low_threshold, high_threshold, blur_ksize=5):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

    # Apply Canny Edge Detector
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    return edges


# Paths to the images
image_path1 = 'data/TestIm1.png'
image_path2 = 'data/TestIm2.png'

# Parameters for Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Apply Canny Edge Detector to both images
edges1 = apply_canny(image_path1, low_threshold, high_threshold)
edges2 = apply_canny(image_path2, low_threshold, high_threshold)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Canny Edges - TestIm1.png')
plt.imshow(edges1, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Canny Edges - TestIm2.png')
plt.imshow(edges2, cmap='gray')
plt.axis('off')

plt.show()
