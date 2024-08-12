


# import numpy as np 
# import cv2
# from picamera2 import Picamera2


# # Open both cameras
# cap_right = Picamera2(1)
# cap_left = Picamera2(0)

# cap_right.start()
# cap_left.start()

# # Reading the mapping values for stereo image rectification


# cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
# Left_Stereo_Map_x = cv_file.getNode('stereoMapL_x').mat()
# Left_Stereo_Map_y = cv_file.getNode('stereoMapL_y').mat()
# Right_Stereo_Map_x = cv_file.getNode('stereoMapR_x').mat()
# Right_Stereo_Map_y = cv_file.getNode('stereoMapR_y').mat()
# cv_file.release()

# disparity = None
# depth_map = None

# # These parameters can vary according to the setup
# max_depth = 400 # maximum distance the setup can measure (in cm)
# min_depth = 20 # minimum distance the setup can measure (in cm)
# depth_thresh = 100.0 # Threshold for SAFE distance (in cm)

# # Reading the stored the StereoBM parameters
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

# # mouse callback function
# def mouse_click(event,x,y,flags,param):
# 	global Z
# 	if event == cv2.EVENT_LBUTTONDBLCLK:
# 		print("Distance = %.2f cm"%depth_map[y,x])	
# 		print("x = %.2f cm"%x)	
# 		print("y = %.2f cm"%y)	


# cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('disp',600,600)
# cv2.setMouseCallback('disp',mouse_click)



# # Creating an object of StereoBM algorithm
# stereo = cv2.StereoBM_create()



# while True:
# 	imgR= cap_right.capture_array()
# 	imgL= cap_left.capture_array()

		
# 	output_canvas = imgL.copy()

# 	imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
# 	imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

# 		# Applying stereo image rectification on the left image
# 	Left_nice= cv2.remap(imgL_gray,
# 							Left_Stereo_Map_x,
# 							Left_Stereo_Map_y,
# 							cv2.INTER_LANCZOS4,
# 							cv2.BORDER_CONSTANT,
# 							0)
		
# 		# Applying stereo image rectification on the right image
# 	Right_nice= cv2.remap(imgR_gray,
# 							Right_Stereo_Map_x,
# 							Right_Stereo_Map_y,
# 							cv2.INTER_LANCZOS4,
# 							cv2.BORDER_CONSTANT,
# 							0)

# 		# Setting the updated parameters before computing disparity map
# 	stereo.setNumDisparities(numDisparities)
# 	stereo.setBlockSize(blockSize)
# 	stereo.setPreFilterType(preFilterType)
# 	stereo.setPreFilterSize(preFilterSize)
# 	stereo.setPreFilterCap(preFilterCap)
# 	stereo.setTextureThreshold(textureThreshold)
# 	stereo.setUniquenessRatio(uniquenessRatio)
# 	stereo.setSpeckleRange(speckleRange)
# 	stereo.setSpeckleWindowSize(speckleWindowSize)
# 	stereo.setDisp12MaxDiff(disp12MaxDiff)
# 	stereo.setMinDisparity(minDisparity)

# 		# Calculating disparity using the StereoBM algorithm
# 	disparity = stereo.compute(Left_nice,Right_nice)
# 		# NOTE: compute returns a 16bit signed single channel image,
# 		# CV_16S containing a disparity map scaled by 16. Hence it 
# 		# is essential to convert it to CV_16S and scale it down 16 times.

# 		# Converting to float32 
# 	disparity = disparity.astype(np.float32)
    
# 	# disparity = disparity/2048
# 		# Normalizing the disparity map
# 	disparity = (disparity/16.0 - minDisparity)/numDisparities
		
# 	depth_map = M/(disparity) # for depth in (cm)

# 	mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
# 	#depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)
# 	# obstacle_avoid()

# 	disparitygray = cv2.cvtColor(disparity, cv2.COLOR_BGR2GRAY)
    
#     # create a kernel for the new image for processing of the depthmap
    
# 	kernel = np.ones((5, 5), np.uint8)

#     # Do some dilation and erosion 
    
# 	dilation = cv2.dilate(disparitygray, kernel, iterations=6)
    
# 	erodila = cv2.erode(dilation, kernel, iterations=6)
    
# 	dilaero = cv2.dilate(erodila, kernel, iterations=17)

    
# 	_, binary_img = cv2.threshold(dilaero, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


#     # Find Contours
    
# 	contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
# 	if contours:
        
# 		largest_contour = max(contours, key=cv2.contourArea)
#         # Draw the largest contour 
        
# 		contour_image = imgR.copy()
        
# 		cv2.drawContours(contour_image,  [largest_contour], -1, (0, 255, 0), 2)

#         # Draw bounding box for the largest contour 
        
# 		x, y, w, h = cv2.boundingRect(largest_contour)
        
# 		cv2.rectangle(contour_image, (x,y), (x+w, y+h), (255, 0, 0), 2)
        
		
# 	cv2.resizeWindow("disp",700,700)
# 	cv2.imshow("disp",contour_image)

# 	if cv2.waitKey(1) == 27:
# 		break
	
# cap_right.stop()
# cap_left.stop()

# cv2.destroyAllWindows()	#depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)



import numpy as np 
import cv2
from picamera2 import Picamera2

# Open both cameras
cap_right = Picamera2(1)
cap_left = Picamera2(0)

cap_right.start()
cap_left.start()

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode('Left_Stereo_Map_x').mat()
Left_Stereo_Map_y = cv_file.getNode('Left_Stereo_Map_y').mat()
Right_Stereo_Map_x = cv_file.getNode('Right_Stereo_Map_x').mat()
Right_Stereo_Map_y = cv_file.getNode('Right_Stereo_Map_y').mat()
cv_file.release()




disparity = None
depth_map = None

# These parameters can vary according to the setup
max_depth = 400 # maximum distance the setup can measure (in cm)
min_depth = 20 # minimum distance the setup can measure (in cm)
depth_thresh = 100.0 # Threshold for SAFE distance (in cm)

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

# Mouse callback function
def mouse_click(event, x, y, flags, param):
    global Z
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("Distance = %.2f cm" % depth_map[y, x])
        print("x = %.2f cm" % x)
        print("y = %.2f cm" % y)

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)
cv2.setMouseCallback('disp', mouse_click)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

while True:
    imgR = cap_right.capture_array()
    imgL = cap_left.capture_array()

    output_canvas = imgL.copy()

    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    # Applying stereo image rectification on the left image
    Left_nice = cv2.remap(imgL_gray,
                          Left_Stereo_Map_x,
                          Left_Stereo_Map_y,
                          cv2.INTER_LANCZOS4,
                          cv2.BORDER_CONSTANT,
                          0)

    # Applying stereo image rectification on the right image
    Right_nice = cv2.remap(imgR_gray,
                           Right_Stereo_Map_x,
                           Right_Stereo_Map_y,
                           cv2.INTER_LANCZOS4,
                           cv2.BORDER_CONSTANT,
                           0)

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

    mask_temp = cv2.inRange(depth_map, min_depth, max_depth)
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

    cv2.resizeWindow("disp", 700, 700)
    cv2.imshow("disp", contour_image)

    if cv2.waitKey(1) == 27:
        break
    
cap_right.stop()
cap_left.stop()

cv2.destroyAllWindows()

