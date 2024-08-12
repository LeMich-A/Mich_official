import sys
import cv2
import numpy as np
import time
import imutils
from picamera2 import Picamera2

# Function for stereo vision and depth estimation
import disparitymap as dismap
import triangulation as tri
import imageprocessor as calibration

# Open both cameras
cap_right = Picamera2(1)
cap_left = Picamera2(0)

cap_right.start()
cap_left.start()

# Stereo vision setup parameters
frame_rate = 120    # Camera frame rate (maximum at 120 fps)
B = 6               # Distance between the cameras [cm]
f = 2.6             # Camera lens's focal length [mm]
alpha = 70          # Camera field of view in the horizontal plane [degrees]

# Main program loop for object detection and depth estimation using stereo vision
while True:
    frame_right = cap_right.capture_array()
    frame_left = cap_left.capture_array()

    ################## CALIBRATION #########################################################
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
    ########################################################################################

    # If cannot catch any frame, break
    if frame_right is None or frame_left is None:
        break

    start = time.time()
    
    # Convert frames to grayscale for disparity calculation
    frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    disparity = stereo.compute(frame_left_gray, frame_right_gray)
    min_val = disparity.min()
    max_val = disparity.max()
    disparity = np.uint8(255 * (disparity - min_val) / (max_val - min_val))

    # Find the closest point (minimum value in the depth map)
    min_loc = cv2.minMaxLoc(disparity)[3]  # This gives the (x, y) of the closest point

    # Create a mask around the closest point for contour detection
    mask = np.zeros(disparity.shape, dtype=np.uint8)
    cv2.circle(mask, min_loc, 20, 255, -1)  # Circle of radius 20 pixels around the closest point

    # Find contours around the closest point
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(frame_right, contours, -1, (0, 255, 0), 2)

    # Display depth information
    depth = 1 / min_val  # This is a simplified depth calculation. You may need to adjust it based on your setup.
    cv2.putText(frame_right, "Depth: " + str(round(depth, 2)) + " units", (min_loc[0], min_loc[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(frame_right, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show the frames and disparity map
    cv2.imshow("Frame Right", frame_right)
    # cv2.imshow("Frame Left", frame_left)
    cv2.imshow("Disparity Map", disparity)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_right.stop()
cap_left.stop()
cv2.destroyAllWindows()
