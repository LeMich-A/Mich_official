import numpy as np 
import cv2
from picamera2 import Picamera2

# Open both cameras
cap_right = Picamera2(1)
cap_left = Picamera2(0)

cap_right.start()
cap_left.start()

cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode('stereoMapL_x').mat()
Left_Stereo_Map_y = cv_file.getNode('stereoMapL_y').mat()
Right_Stereo_Map_x = cv_file.getNode('stereoMapR_x').mat()
Right_Stereo_Map_y = cv_file.getNode('stereoMapR_y').mat()

cv_file.release()


def nothing(x):
    pass

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)

cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

stereo = cv2.StereoBM_create()
disparity = None
depth_map = None

max_depth = 400
min_depth = 20
depth_thresh = 100.0

# def mouse_click(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print("Distance = %.2f cm" % depth_map[y, x])

# cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('disp', 600, 600)
# cv2.setMouseCallback('disp', mouse_click)

output_canvas = None

def obstacle_avoid():
    global output_canvas
    mask = cv2.inRange(depth_map, 10, depth_thresh)

    if np.sum(mask) / 255.0 > 0.01 * mask.shape[0] * mask.shape[1]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)

        if cv2.contourArea(cnts[0]) > 0.01 * mask.shape[0] * mask.shape[1]:
            x, y, w, h = cv2.boundingRect(cnts[0])
            mask2 = np.zeros_like(mask)
            cv2.drawContours(mask2, cnts, 0, (255), -1)
            depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
            cv2.putText(output_canvas, "WARNING !", (x+5, y-40), 1, 2, (0, 0, 255), 2, 2)
            cv2.putText(output_canvas, "Object at", (x+5, y), 1, 2, (100, 10, 25), 2, 2)
            cv2.putText(output_canvas, "%.2f cm" % depth_mean, (x+5, y+40), 1, 2, (100, 10, 25), 2, 2)
    else:
        cv2.putText(output_canvas, "SAFE!", (100, 100), 1, 3, (0, 255, 0), 2, 3)

    cv2.imshow('output_canvas', output_canvas)

while True:
    imgL = cap_right.capture_array()
    imgR = cap_left.capture_array()

    if imgL is None or imgR is None:
        print("Failed to capture images")
        continue

    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    Left_nice = cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    numDisparities = max(cv2.getTrackbarPos('numDisparities', 'disp') * 16, 16)
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')
    M = cv_file.getNode("M").real()
    
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    disparity = stereo.compute(Left_nice, Right_nice)
    disparity = disparity.astype(np.float32)
    disparity = (disparity / 16.0 - minDisparity) / numDisparities

    with np.errstate(divide='ignore'):
        depth_map = M / disparity
        depth_map[disparity == 0] = np.inf

    mask_temp = cv2.inRange(depth_map, min_depth, max_depth)
    depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask_temp)

    if output_canvas is None:
        output_canvas = np.zeros_like(Left_nice)

    obstacle_avoid()

    cv2.resizeWindow("disp", 700, 700)
    cv2.imshow("disp", disparity)

    if cv2.waitKey(1) == 27:
        break

cap_right.stop()
cap_left.stop()


print("Saving depth estimation parameters ......")

cv_file = cv2.FileStorage("../data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("numDisparities",numDisparities)
cv_file.write("blockSize",blockSize)
cv_file.write("preFilterType",preFilterType)
cv_file.write("preFilterSize",preFilterSize)
cv_file.write("preFilterCap",preFilterCap)
cv_file.write("textureThreshold",textureThreshold)
cv_file.write("uniquenessRatio",uniquenessRatio)
cv_file.write("speckleRange",speckleRange)
cv_file.write("speckleWindowSize",speckleWindowSize)
cv_file.write("disp12MaxDiff",disp12MaxDiff)
cv_file.write("minDisparity",minDisparity)
cv_file.write("M",39.075)
cv_file.release()

cv2.destroyAllWindows()
