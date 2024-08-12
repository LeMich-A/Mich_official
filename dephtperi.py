# from picamera2 import Picamera2
# import cv2
# import numpy as np
# # Function for stereo vision and depth estimation


# import imageprocessor as calibration

# # Mediapipe for face detection
# import mediapipe as mp
# import time

# cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
# Left_Stereo_Map_x = cv_file.getNode('stereoMapL_x').mat()
# Left_Stereo_Map_y = cv_file.getNode('stereoMapL_y').mat()
# Right_Stereo_Map_x = cv_file.getNode('stereoMapR_x').mat()
# Right_Stereo_Map_y = cv_file.getNode('stereoMapR_y').mat()
# cv_file.release()

# # Parameters for depth measurement
# max_depth = 400  # maximum distance (cm)
# min_depth = 20   # minimum distance (cm)
# depth_thresh = 100.0  # threshold for safe distance (cm)

# # Reading the stored StereoBM parameters
# cv_file = cv2.FileStorage("../data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
# numDisparities = int(cv_file.getNode("numDisparities").real())
# blockSize = int(cv_file.getNode("blockSize").real())
# preFilterType = int(cv_file.getNode("preFilterType").real())
# preFilterSize = int(cv_file.getNode("preFilterSize").real())
# preFilterCap = int(cv_file.getNode("preFilterCap").real())
# textureThreshold = int(cv_file.getNode("textureThreshold").real())
# uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
# speckleRange = int(cv_file.getNode("speckleRange").real())
# speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
# disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
# minDisparity = int(cv_file.getNode("minDisparity").real())
# M = cv_file.getNode("M").real()
# cv_file.release()

# stereo = cv2.StereoBM_create()

# def findobject (rightimg , leftimage) :
#     # replace the ROI by imgR and imgL capture from the perimeter_detect 
#     imgR = rightimg 
#     imgL = leftimage
#     output_canvas = imgL.copy()

#     imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
#     imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

#     # Applying stereo image rectification
#     Left_nice = cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
#     Right_nice = cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
#     # Setting the updated parameters before computing disparity map
#     stereo.setNumDisparities(numDisparities)
#     stereo.setBlockSize(blockSize)
#     stereo.setPreFilterType(preFilterType)
#     stereo.setPreFilterSize(preFilterSize)
#     stereo.setPreFilterCap(preFilterCap)
#     stereo.setTextureThreshold(textureThreshold)
#     stereo.setUniquenessRatio(uniquenessRatio)
#     stereo.setSpeckleRange(speckleRange)
#     stereo.setSpeckleWindowSize(speckleWindowSize)
#     stereo.setDisp12MaxDiff(disp12MaxDiff)
#     stereo.setMinDisparity(minDisparity)

#     # Calculating disparity using the StereoBM algorithm
#     disparity = stereo.compute(Left_nice, Right_nice)
#     disparity = disparity.astype(np.float32)
#     # disparity = disparity / 2048.0

#     # Normalizing the disparity map
#     disparity = (disparity / 16.0 - minDisparity) / numDisparities
    
#     depth_map = M / disparity  # for depth in (cm)

#     mask_temp = cv2.inRange(depth_map, min_depth, max_depth)


#     disparitygray = disparity  # Skip the color conversion as disparity is already a single-channel image
    
#     # Create a kernel for the new image for processing of the depthmap
#     kernel = np.ones((5, 5), np.uint8)

#     # Do some dilation and erosion 
#     dilation = cv2.dilate(disparitygray, kernel, iterations=6)
#     erodila = cv2.erode(dilation, kernel, iterations=6)
#     dilaero = cv2.dilate(erodila, kernel, iterations=17)

#     # Normalize and convert to 8-bit for thresholding
#     dilaero_norm = cv2.normalize(dilaero, None, 0, 255, cv2.NORM_MINMAX)
#     dilaero_8u = cv2.convertScaleAbs(dilaero_norm)

#     _, binary_img = cv2.threshold(dilaero_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Find Contours
#     contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         # Draw the largest contour 
#         contour_image = imgR.copy()
#         cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

#         # Draw bounding box for the largest contour 
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        



#     return contour_image



import cv2
import numpy as np
from picamera2 import Picamera2

# Function for stereo vision and depth estimation


# Parameters for depth measurement
max_depth = 400  # maximum distance (cm)
min_depth = 20   # minimum distance (cm)
depth_thresh = 100.0  # threshold for safe distance (cm)

# Load the stereo rectification maps
cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode('stereoMapL_x').mat()
Left_Stereo_Map_y = cv_file.getNode('stereoMapL_y').mat()
Right_Stereo_Map_x = cv_file.getNode('stereoMapR_x').mat()
Right_Stereo_Map_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()


# Check if the maps are loaded correctly
if Left_Stereo_Map_x is None or Left_Stereo_Map_y is None or Right_Stereo_Map_x is None or Right_Stereo_Map_y is None:
    raise ValueError("One or more stereo rectification maps are not loaded correctly.")

# Reading the stored StereoBM parameters
cv_file = cv2.FileStorage("../data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
preFilterType = int(cv_file.getNode("preFilterType").real())
preFilterSize = int(cv_file.getNode("preFilterSize").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
textureThreshold = int(cv_file.getNode("textureThreshold").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
minDisparity = int(cv_file.getNode("minDisparity").real())
M = cv_file.getNode("M").real()
cv_file.release()

stereo = cv2.StereoBM_create()

def findobject(rightimg, leftimage):
    imgR = rightimg 
    imgL = leftimage
    output_canvas = imgL.copy()

    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    # Applying stereo image rectification
    Left_nice = cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    # Setting the updated parameters before computing disparity map
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

    # # Calculating disparity using the StereoBM algorithm
    # disparity = stereo.compute(Left_nice, Right_nice)
    # disparity = disparity.astype(np.float32)

    # # Normalizing the disparity map
    # disparity = (disparity / 16.0 - minDisparity) / numDisparities
    
    # depth_map = M / disparity  # for depth in (cm)

    # mask_temp = cv2.inRange(depth_map, min_depth, max_depth)

    # disparitygray = disparity  # Skip the color conversion as disparity is already a single-channel image
    
    # # Create a kernel for the new image for processing of the depthmap
    # kernel = np.ones((5, 5), np.uint8)

    # # Do some dilation and erosion 
    # dilation = cv2.dilate(disparitygray, kernel, iterations=6)
    # erodila = cv2.erode(dilation, kernel, iterations=6)
    # dilaero = cv2.dilate(erodila, kernel, iterations=17)

    # # Normalize and convert to 8-bit for thresholding
    # dilaero_norm = cv2.normalize(dilaero, None, 0, 255, cv2.NORM_MINMAX)
    # dilaero_8u = cv2.convertScaleAbs(dilaero_norm)

    # _, binary_img = cv2.threshold(dilaero_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Find Contours
    # contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if contours:
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     # Draw the largest contour 
    #     contour_image = imgR.copy()
    #     cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

    #     # Draw bounding box for the largest contour 
    #     x, y, w, h = cv2.boundingRect(largest_contour)
    #     cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice, Right_nice)
    # NOTE: compute returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_16S and scale it down 16 times.

    # Converting to float32 
    disparity = disparity.astype(np.float32)
    
    # disparity = disparity / 2048
    # Normalizing the disparity map
    disparity = (disparity / 16.0 - minDisparity) / numDisparities
        
    depth_map = M / disparity  # for depth in (cm)

    # mask_temp = cv2.inRange(depth_map, min_depth, max_depth)
    # depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask_temp)
    # obstacle_avoid()

    disparitygray = disparity  # Skip the color conversion as disparity is already a single-channel image
    
    # Create a kernel for the new image for processing of the depthmap
    kernel = np.ones((5, 5), np.uint8)

    # Do some dilation and erosion 
    dilation = cv2.dilate(disparitygray, kernel, iterations=6)
    erodila = cv2.erode(dilation, kernel, iterations=6)
    dilaero = cv2.dilate(erodila, kernel, iterations=17)

    # Normalize and convert to 8-bit for thresholding
    dilaero_norm = cv2.normalize(dilaero, None, 0, 255, cv2.NORM_MINMAX)
    dilaero_8u = cv2.convertScaleAbs(dilaero_norm)

    _, binary_img = cv2.threshold(dilaero_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find Contours
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw the largest contour 
        contour_image = imgR.copy()
        cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

        # Draw bounding box for the largest contour 
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)


    return contour_image , binary_img


cv2.destroyAllWindows()