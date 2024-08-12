# import cv2
# import numpy as np
# from picamera2 import Picamera2

# # Global variables
# roi_selected = False
# roi = (0, 0, 0, 0)  # x, y, w, h
# start_point = None





# def select_roi(event, x, y, flags, param):
#     global roi, roi_selected, start_point
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         start_point = (x, y)
#         roi_selected = False

#     elif event == cv2.EVENT_MOUSEMOVE and start_point is not None:
#         # Draw rectangle while moving the mouse
#         frame_temp = frame.copy()
#         cv2.rectangle(frame_temp, start_point, (x, y), (0, 255, 0), 2)
#         cv2.imshow('Camera', frame_temp)

#     elif event == cv2.EVENT_LBUTTONUP:
#         roi_selected = True
#         roi = (start_point[0], start_point[1], x - start_point[0], y - start_point[1])
#         start_point = None

# # Open the camera
# cap = Picamera2(1)    
# cap.start()


# # Create a window and set the mouse callback function
# cv2.namedWindow('Camera')
# cv2.setMouseCallback('Camera', select_roi)

# while True:
#     frame = cap.capture_array()
    

#     if roi_selected:
#         x, y, w, h = roi
#         roi_frame = frame[y:y+h, x:x+w]

#         # Convert ROI to grayscale and find contours
#         gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
#         blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
#         edged_roi = cv2.Canny(blurred_roi, 50, 150)
        
#         contours, _ = cv2.findContours(edged_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Draw contours on the original frame
#         for contour in contours:
#             contour += np.array([x, y])  # Shift contour to the original frame coordinates
#             cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

#         # Draw ROI rectangle
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     cv2.imshow('Camera', frame)

#     key = cv2.waitKey(1)
#     if key == 27:  # ESC key to break
#         break

# # Release the camera and close all windows
# cap.stop()

# cv2.destroyAllWindows()

import cv2
import numpy as np
from picamera2 import Picamera2

# Global variables
roi_selected = False
roi = (0, 0, 0, 0)  # x, y, w, h
start_point = None

def select_roi(event, x, y, flags, param):
    global roi, roi_selected, start_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        roi_selected = False

    elif event == cv2.EVENT_MOUSEMOVE and start_point is not None:
        # Draw rectangle while moving the mouse
        frame_temp = frame.copy()
        cv2.rectangle(frame_temp, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow('Camera', frame_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_selected = True
        roi = (start_point[0], start_point[1], x - start_point[0], y - start_point[1])
        start_point = None

# Open the camera
cap = Picamera2(1)    
cap.start()

# Create a window and set the mouse callback function
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', select_roi)

while True:
    frame = cap.capture_array()
    
    if roi_selected:
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]

        # Convert ROI to grayscale and find contours
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Apply mean filter
        kernelMean = np.ones((31,31), np.float32) / 961
        imgMean = cv2.filter2D(gray_roi, -1, kernelMean)

         # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Apply thresholding to obtain binary edges
        threshold = 110
        edges = np.uint8(gradient_magnitude > threshold) * 255

    # Convert edges to binary image
        ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Compute the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Update ROI position to follow the centroid
                x = x + cX - w // 2
                y = y + cY - h // 2
                roi = (x, y, w, h)
                
                # Ensure ROI is within frame bounds
                x = max(0, min(x, frame.shape[1] - w))
                y = max(0, min(y, frame.shape[0] - h))
                roi = (x, y, w, h)
                
            # Shift contour to the original frame coordinates
            largest_contour += np.array([x, y])
            
            # Draw the largest contour on the original frame
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        cv2.imshow('edges', binary_edges)
        # Draw ROI rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Camera', frame)
    

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

# Release the camera and close all windows
cap.stop()
cv2.destroyAllWindows()
