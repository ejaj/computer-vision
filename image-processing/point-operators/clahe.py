import cv2
import matplotlib.pyplot as plt

image = cv2.imread("/home/kazi/Works/Dtu/computer-vision/data/lena.png", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not loaded. Please check the file path.")
    exit()

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')  # Hide axis
plt.subplot(1, 2, 2)
plt.title("CLAHE Image")
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')  # Hide axis
plt.show()
