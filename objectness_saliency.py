import cv2
import numpy as np

# Load the image
image_path = "data/naymer.jpg"
image = cv2.imread(image_path)

# Initialize the ObjectnessBING saliency detector
saliency = cv2.saliency.ObjectnessBING_create()

# Set the training path for ObjectnessBING (required)
saliency.setTrainingPath("data/saliency/samples/ObjectnessTrainedModel")

# Compute the bounding box predictions for saliency
success, saliencyMap = saliency.computeSaliency(image)

# Check if saliencyMap is valid and contains entries
if success and saliencyMap is not None and len(saliencyMap) > 0:
    numDetections = saliencyMap.shape[0]
    
    # Draw each bounding box on the image
    output = image.copy()
    for i in range(numDetections):
        # Check if each row has exactly two points (two sets of coordinates)
        if len(saliencyMap[i]) == 1 and len(saliencyMap[i][0]) == 4:
            # Extract coordinates for the top-left and bottom-right corners
            x1, y1, x2, y2 = saliencyMap[i][0]
            
            # Draw the rectangle on the image
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        else:
            print(f"Skipping invalid saliency entry at index {i}: {saliencyMap[i]}")

    # Display the result
    cv2.imshow("Saliency Detection with ObjectnessBING", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No valid saliency map or detections.")
