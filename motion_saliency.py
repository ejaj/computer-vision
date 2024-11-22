from imutils.video import VideoStream
import imutils
import time
import cv2

# Initialize the motion saliency object and start the video stream
saliency = None
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to 500px
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    
    # Initialize the saliency object if it hasn't been done yet
    if saliency is None:
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        saliency.init()

    # Convert the frame to grayscale as required by the saliency detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the saliency map
    success, saliencyMap = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # Display the original frame and the saliency map
    cv2.imshow("Frame", frame)
    cv2.imshow("Map", saliencyMap)
    
    # Check for key press; if 'q' is pressed, exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
