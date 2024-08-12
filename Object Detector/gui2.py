import numpy as np 
import cv2
from picamera2 import Picamera2

# Initialize Pi Camera 2
cap_right = Picamera2(1)
cap_left = Picamera2(0)

cap_right.start()
cap_left.start()


# Create a StereoBM object
stereo_bm = cv2.StereoBM_create()

# Initialize trackbars for parameters
min_disparity = 0
max_disparity = 16
block_size = 15
pre_filter_cap = 61
uniqueness_ratio = 10
speckle_window_size = 100
speckle_range = 32
disp12_max_diff = 1

cv2.namedWindow("Depth Map GUI")
cv2.createTrackbar("Min Disparity", "Depth Map GUI", min_disparity, max_disparity, lambda x: None)
cv2.createTrackbar("Block Size", "Depth Map GUI", block_size, 255, lambda x: None)
cv2.createTrackbar("Pre-Filter Cap", "Depth Map GUI", pre_filter_cap, 255, lambda x: None)
cv2.createTrackbar("Uniqueness Ratio", "Depth Map GUI", uniqueness_ratio, 100, lambda x: None)
cv2.createTrackbar("Speckle Window Size", "Depth Map GUI", speckle_window_size, 255, lambda x: None)
cv2.createTrackbar("Speckle Range", "Depth Map GUI", speckle_range, 255, lambda x: None)
cv2.createTrackbar("Disp12 Max Diff", "Depth Map GUI", disp12_max_diff, 10, lambda x: None)

while True:
    # Capture images from Pi Camera 2
    left_image = cap_left.capture_array()
    right_image = cap_right.capture_array()

    # Convert images to grayscale
    left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

       # Update parameters from trackbars
    min_disparity = cv2.getTrackbarPos("Min Disparity", "Depth Map GUI")
    max_disparity = cv2.getTrackbarPos("Max Disparity", "Depth Map GUI")
    max_disparity = (max_disparity // 16) * 16  # Ensure max_disparity is divisible by 16

    if min_disparity >= max_disparity:
        min_disparity = max_disparity - 1

    stereo_bm.setMinDisparity(min_disparity)
    stereo_bm.setNumDisparities(max_disparity)
   
    block_size = cv2.getTrackbarPos("Block Size", "Depth Map GUI")
    pre_filter_cap = cv2.getTrackbarPos("Pre-Filter Cap", "Depth Map GUI")
    uniqueness_ratio = cv2.getTrackbarPos("Uniqueness Ratio", "Depth Map GUI")
    speckle_window_size = cv2.getTrackbarPos("Speckle Window Size", "Depth Map GUI")
    speckle_range = cv2.getTrackbarPos("Speckle Range", "Depth Map GUI")
    disp12_max_diff = cv2.getTrackbarPos("Disp12 Max Diff", "Depth Map GUI")

    stereo_bm.setPreFilterCap(pre_filter_cap)
    stereo_bm.setBlockSize(block_size)
    stereo_bm.setMinDisparity(min_disparity)
    stereo_bm.setNumDisparities(max_disparity)
    stereo_bm.setUniquenessRatio(uniqueness_ratio)
    stereo_bm.setSpeckleWindowSize(speckle_window_size)
    stereo_bm.setSpeckleRange(speckle_range)
    stereo_bm.setDisp12MaxDiff(disp12_max_diff)

    # Compute depth map
    disparity = stereo_bm.compute(left_image_gray, right_image_gray)

    # Normalize disparity map for display
    normalized_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

    # Display depth map
    cv2.imshow("Depth Map GUI", normalized_disparity)

    # Exit on key press
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()