import cv2
import matplotlib.pyplot as plt


def apply_canny(image_path, low_threshold, high_threshold, blur_ksize=5):
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


# Path to the image
image_path = 'data/TestIm2.png'

# Different sets of parameters for Canny Edge Detector
params = [
    (50, 150),
    (100, 200),
    (150, 250),
    (200, 300),
    (50, 100),
    (30, 90)
]

# Apply Canny Edge Detector with different parameters
results = [apply_canny(image_path, low, high) for low, high in params]

# Display the results
plt.figure(figsize=(18, 12))

for i, (edges, (low, high)) in enumerate(zip(results, params)):
    plt.subplot(2, 3, i + 1)
    plt.title(f'Canny Edges\nLow: {low}, High: {high}')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

plt.show()
