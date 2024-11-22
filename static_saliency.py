import cv2

image_path = "data/naymer.jpg"
image = cv2.imread(image_path)

# saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
# (success, saliencyMap) = saliency.computeSaliency(image)
# saliencyMap = (saliencyMap * 255).astype("uint8")
# cv2.imshow("Image", image)
# cv2.imshow("Output", saliencyMap)
# cv2.waitKey(0)

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)
# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map
saliencyMap = (saliencyMap * 255).astype("uint8")

threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# show the images
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.imshow("Thresh", threshMap)
cv2.waitKey(0)