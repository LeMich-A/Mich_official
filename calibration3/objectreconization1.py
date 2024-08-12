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

def detect_object(frame, color_range_lower, color_range_upper, min_area):
    """Detect objects based on color range and return the bounding box and center point."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, color_range_lower, color_range_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                return box, center
    return None, None

# Define color range for object detection (e.g., red color)
color_range_lower = np.array([0, 120, 70])
color_range_upper = np.array([10, 255, 255])

# Minimum area for detected objects
min_area = 500  # Adjust this value based on the size of objects you want to detect

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

    box_right, center_point_right = detect_object(frame_right, color_range_lower, color_range_upper, min_area)
    box_left, center_point_left = detect_object(frame_left, color_range_lower, color_range_upper, min_area)

    # If no object is detected in one of the cameras, show text "TRACKING LOST"
    if center_point_right is None or center_point_left is None:
        cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Function to calculate depth of object. Outputs vector of all depths in case of several faces.
        depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

        cv2.putText(frame_right, "Distance: " + str(round(depth, 1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame_left, "Distance: " + str(round(depth, 1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        print("Depth: ", str(round(depth, 1)))

    # Convert frames to grayscale for disparity calculation
    frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    disparity = stereo.compute(frame_left_gray, frame_right_gray)
    min_val = disparity.min()
    max_val = disparity.max()
    disparity = np.uint8(255 * (disparity - min_val) / (max_val - min_val))

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(frame_right, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame_left, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

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

