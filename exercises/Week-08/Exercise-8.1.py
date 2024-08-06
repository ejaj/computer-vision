import cv2
import matplotlib.pyplot as plt

image_url = 'data/sunflowers.jpg'
image = cv2.imread(image_url)

# Convert the image to grayscale and then to float
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

# Display the grayscale image
plt.imshow(image_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()


def scale_spaced(im, sigma, n):
    im_scales = []
    for i in range(n):
        sigma_i = sigma * (2 ** i)
        blurred_im = cv2.GaussianBlur(im, (0, 0), sigma_i)
        im_scales.append(blurred_im)
    return im_scales


sigma = 1.0
n = 5
im_scales = scale_spaced(image, sigma, n)

plt.figure(figsize=(15, 5))
for i, img in enumerate(im_scales):
    plt.subplot(1, n, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'Scaled Image {i + 1}')
plt.show()
