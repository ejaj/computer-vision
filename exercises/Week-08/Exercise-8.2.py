import cv2
import matplotlib.pyplot as plt


def scale_spaced(im, sigma, n):
    im_scales = []
    for i in range(n):
        sigma_i = sigma * (2 ** i)
        blurred_im = cv2.GaussianBlur(im, (0, 0), sigma_i)
        im_scales.append(blurred_im)
    return im_scales


def differenceOfGaussians(im, sigma, n):
    im_scales = scale_spaced(im, sigma, n)
    DoG = []
    for i in range(n - 1):
        dog = im_scales[i + 1] - im_scales[i]
        DoG.append(dog)
    return DoG


img = cv2.imread('data/sunflowers.jpg')
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
plt.imshow(image_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Create the scale space DoGs
sigma = 1.0
n = 5
DoG = differenceOfGaussians(image_gray, sigma, n)

# Display the DoG images
plt.figure(figsize=(15, 5))
for i, dog in enumerate(DoG):
    plt.subplot(1, n - 1, i + 1)
    plt.imshow(dog, cmap='gray')
    plt.title(f'DoG {i}')
    plt.axis('off')
plt.show()
