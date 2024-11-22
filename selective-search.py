import random
import time
import cv2

image_path = "data/dog.jpg"
image = cv2.imread(image_path)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# Fast
# ss.switchToSelectiveSearchFast()

# slower/quality
ss.switchToSelectiveSearchQuality()

start = time.time()
rects = ss.process()
end = time.time()
print("[INFO] selective search took {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(rects)))

# visualize
# for i in range(0, len(rects), 100):
# 	# clone the original image so we can draw on it
# 	output = image.copy()
# 	# loop over the current subset of region proposals
# 	for (x, y, w, h) in rects[i:i + 100]:
# 		# draw the region proposal bounding box on the image
# 		color = [random.randint(0, 255) for j in range(0, 3)]
# 		cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
# 	# show the output image
# 	cv2.imshow("Output", output)
# 	key = cv2.waitKey(0) & 0xFF
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break

scores = [random.uniform(0.5, 1.0) for _ in range(len(rects))]  # Confidence scores between 0.5 and 1.0

# Convert rects to the format required by cv2.dnn.NMSBoxes: [x, y, w, h]
boxes = [[x, y, w, h] for (x, y, w, h) in rects]

# Apply Non-Maxima Suppression
conf_threshold = 0.6  # Only consider boxes above this confidence
nms_threshold = 0.4   # IoU threshold for NMS
indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

# Check if indices is empty or not
if len(indices) > 0:
    # Visualize NMS-filtered boxes
    output = image.copy()
    for i in indices:
        # Use `i` as an index directly
        i = i[0] if isinstance(i, (list, tuple)) else i
        (x, y, w, h) = boxes[i]
        confidence = scores[i]
        
        # Draw the bounding box
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Display the confidence score
        label = f"{confidence:.2f}"
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"Box {i}: (x={x}, y={y}, w={w}, h={h}), Confidence={confidence:.2f}")
    # Show the output image with NMS-filtered boxes and confidence scores
    cv2.imshow("NMS Output with Confidence", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[INFO] No boxes remained after NMS.")