import cv2
import imutils
from skimage.transform import pyramid_gaussian
import time

def pyramid_opencv(image, scale=1.5, minSize=(30, 30)):
    yield image
    while True:
        # Compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # Stop if the resized image does not meet the minimum size
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

def pyramid_skimage(image, downscale=2):
   for i, resized in enumerate(pyramid_gaussian(image, downscale=downscale)):
        # Stop if the image size becomes too small
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
        yield resized

def sliding_window(image, step, window_size):
    # Slide a window across the image
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

        
scale=1.5
image_path = "data/kazi.jpg"
image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

window_size = (128, 128)  
step_size = 64  

# Loop over each layer of the pyramid
for layer_num, layer in enumerate(pyramid_opencv(image, scale=1.5)):
    print(f"Processing layer {layer_num + 1} with shape {layer.shape}")

    # Loop over each sliding window in the current layer
    for (x, y, window) in sliding_window(layer, step=step_size, window_size=window_size):
        # Check if the window meets the desired window size
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue

        # Draw the window rectangle on the layer copy
        clone = layer.copy()
        cv2.rectangle(clone, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)

        # Display the current window on the image
        cv2.imshow(f"Pyramid Layer {layer_num + 1}", clone)
        cv2.waitKey(1)
        time.sleep(0.03)  # Pause briefly to visualize each window

    # After processing each layer, close the window
    cv2.destroyAllWindows()



# # Display each layer in the pyramid using OpenCV
# print("Pyramid with OpenCV and imutils:")
# opencv_layer_count = 0
# for (i, resized) in enumerate(pyramid_opencv(image, scale=scale)):
#     opencv_layer_count += 1
#     cv2.imshow(f"OpenCV Layer {i + 1}", resized)
#     cv2.waitKey(0)

# print(f"Total layers in OpenCV pyramid: {opencv_layer_count}")


# Count layers in scikit-image pyramid
# skimage_layer_count = 0
# print("Pyramid with scikit-image:")
# for i, resized in enumerate(pyramid_skimage(image_rgb, downscale=2)):
#     # Convert the float image to uint8 format
#     resized_uint8 = (resized * 255).astype("uint8")

#     # Handle unexpected channel shapes by forcing RGB conversion
#     if len(resized_uint8.shape) == 2:  # Grayscale image, convert to BGR
#         resized_bgr = cv2.cvtColor(resized_uint8, cv2.COLOR_GRAY2BGR)
#     elif len(resized_uint8.shape) == 3 and resized_uint8.shape[2] == 2:  # Two channels, add a third channel
#         resized_bgr = cv2.merge([resized_uint8, resized_uint8[:, :, 1]])  # Duplicate one channel to get RGB
#     elif len(resized_uint8.shape) == 3 and resized_uint8.shape[2] == 3:  # RGB image
#         resized_bgr = cv2.cvtColor(resized_uint8, cv2.COLOR_RGB2BGR)
#     else:
#         print(f"Skipping layer with unexpected number of channels: {resized_uint8.shape}")
#         continue

#     skimage_layer_count += 1
#     cv2.imshow(f"scikit-image Layer {skimage_layer_count}", resized_bgr)
#     cv2.waitKey(0)  
# print(f"Total layers in scikit-image pyramid: {skimage_layer_count}")

# cv2.destroyAllWindows()