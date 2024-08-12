import numpy as np
import cv2
from picamera2 import Picamera2
import time

import triangulation as tri
import imageprocessor as calibration

# Open both cameras
cap_right = Picamera2(1)                 
cap_left = Picamera2(0)

cap_right.start()
cap_left.start()


class UserRect():
    def __init__(self) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        
selectStereo = UserRect()
followStereo = UserRect()

def on_mouseStereo(event, x, y, flags, param):
    global selectStereo, followStereo
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        selectStereo.start_x = x - 4 if x - 4 > 0 else 0
        selectStereo.start_y = y - 4 if y - 4 > 0 else 0
        selectStereo.end_x = x + 4 if x + 4 < 240 else 240
        selectStereo.end_y = y + 4 if y + 4 < 180 else 180
    else:
        followStereo.start_x = x - 4 if x - 4 > 0 else 0
        followStereo.start_y = y - 4 if y - 4 > 0 else 0
        followStereo.end_x = x + 4 if x + 4 < 240 else 240
        followStereo.end_y = y + 4 if y + 4 < 180 else 180

cv2.namedWindow("liveDepthViewColour", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("liveDepthViewColour", on_mouseStereo)

# Main loop
while True:
    frame_right = cap_right.capture_array()
    frame_left = cap_left.capture_array()
    
    # Calibration
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    # Drawing rectangle
    cv2.rectangle(frame_left, (selectStereo.start_x, selectStereo.start_y), 
                  (selectStereo.end_x, selectStereo.end_y), (128, 128, 128), 1)
    
    cv2.imshow("liveDepthViewColour", frame_left)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.stop()
cap_left.stop()
cv2.destroyAllWindows()
