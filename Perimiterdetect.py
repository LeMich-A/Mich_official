# /////////////////////////////////// FIRST SCRIPT //////////////////////////////////////////////////

# import cv2
# import numpy as np
# import math
# import time 
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# response_times = []
# iterations = []
# def detect_and_draw_circles(image_path):
#     # Load the image
#     img = cv2.imread(image_path)

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Apply mean filter
#     # kernelMean = np.ones((31,31), np.float32) / 961
#     # imgMean = cv2.filter2D(gray, -1, kernelMean)

#     # Apply Sobel edge detection
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

#     # Apply thresholding to obtain binary edges
#     threshold = 110
#     edges = np.uint8(gradient_magnitude > threshold) * 255

#     # Convert edges to binary image
#     ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

    

   
#     messages = []
#      # Record the start time
#     start_time = time.time()
#     # Define parameter values
#     param1_values = [200]  # Upper threshold for the edge detector
#     param2_values = [38]   # Threshold for circle detection
#     dp_values = [1]        # Inverse ratio of the accumulator resolution to the image resolution

#     best_params = None
#     best_circles = None
#     best_circle_count = 0

# # Calculate minDist value
#     minDist_value = img.shape[0] // 4

# # Iterate over parameter combinations
#     for dp in dp_values:
#         for param1 in param1_values:
#             for param2 in param2_values:
#                 circles = cv2.HoughCircles(binary_edges,
#                                        cv2.HOUGH_GRADIENT,
#                                        dp=dp,
#                                        minDist=minDist_value,
#                                        param1=param1,
#                                        param2=param2,
#                                        minRadius=0,
#                                        maxRadius=0)
#                 if circles is not None:
#                     circle_count = len(circles[0])
#                     if circle_count > best_circle_count:
#                         best_circle_count = circle_count
                        
#                         best_circles = circles
#                         messages.append("Object within workspace.")

#                 else:
#                         messages.append("Overhanging")
    
#         if best_circles is not None:
#             best_circles = np.uint16(np.around(best_circles))
#             for circle in best_circles[0, :1]:
#                 center = (circle[0], circle[1])
#                 radius = circle[2]
#         # Draw circle
#                 cv2.circle(rgb_img, center, radius, (0, 255, 0), 2)

#     end_time = time.time()

#     response_time = end_time - start_time

#       # Append response time and iteration number
#     response_times.append(response_time)
#     iterations.append(len(iterations) + 1)

#     # Plot live response time
#     plt.clf()
#     plt.plot(iterations, response_times, marker='o')
#     plt.xlabel('Iteration')
#     plt.ylabel('Response Time (s)')
#     plt.title('Response Time')
#     plt.grid(True)
#     plt.pause(0.001)  # Pause to allow the plot to update

    

#     return rgb_img,messages


# /////////////////////////////////// THE END /////////////////////////////////////////////////



# # /////////////////////////////////////////////////// SCRIPT, PLOT AND  WITH THE THE MASK /////////////////////////////////////////////

# import cv2
# import numpy as np
# import math
# import time 
# # import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation

# response_times = []
# iterations = []

# def detect_and_draw_circles(image_path):
#     # Load the image
#     img = cv2.imread(image_path)

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Apply Sobel edge detection
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

#     # Apply thresholding to obtain binary edges
#     threshold = 110
#     edges = np.uint8(gradient_magnitude > threshold) * 255

#     # Convert edges to binary image
#     ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

#     message = ""
#     # Record the start time
#     start_time = time.time()
#     # Define parameter values
#     param1_values = [200]  # Upper threshold for the edge detector
#     param2_values = [38]   # Threshold for circle detection
#     dp_values = [1]        # Inverse ratio of the accumulator resolution to the image resolution

#     best_params = None
#     best_circles = None
#     best_circle_count = 0

#     # Calculate minDist value
#     minDist_value = img.shape[0] // 4

#     # Iterate over parameter combinations
#     for dp in dp_values:
#         for param1 in param1_values:
#             for param2 in param2_values:
#                 circles = cv2.HoughCircles(binary_edges,
#                                            cv2.HOUGH_GRADIENT,
#                                            dp=dp,
#                                            minDist=minDist_value,
#                                            param1=param1,
#                                            param2=param2,
#                                            minRadius=0,
#                                            maxRadius=0)
#                 if circles is not None:
#                     circle_count = len(circles[0])
#                     if circle_count > best_circle_count:
#                         best_circle_count = circle_count
#                         best_circles = circles
#                         message = "Object within workspace."
#                 else:
#                     message = "Overhanging"

#     if best_circles is not None:
#         best_circles = np.uint16(np.around(best_circles))
#         for circle in best_circles[0, :1]:
#             center = (circle[0], circle[1])
#             radius = circle[2]
#             # Draw circle
#             cv2.circle(rgb_img, center, radius, (0, 255, 0), 2)

#             # Create a mask for the region inside of the ROI
#             mask = np.zeros_like(img)
#             cv2.circle(mask, center, radius - 40, (255, 255, 255), thickness=-1)  # Fill the circle in the mask

#             # Set pixels outside of the ROI to a dark color (e.g., black)
#             rgb_img[mask[:, :, 0] == 0] = [0, 0, 0]
    
#     end_time = time.time()

#     response_time = end_time - start_time

#     # Append response time and iteration number
#     response_times.append(response_time)
#     iterations.append(len(iterations) + 1)

#     # Plot live response time
#     # plt.clf()
#     # plt.plot(iterations, response_times, marker='o')
#     # plt.xlabel('Iteration')
#     # plt.ylabel('Response Time (s)')
#     # plt.title('Response Time')
#     # plt.grid(True)
#     # plt.pause(0.001)  # Pause to allow the plot to update

#     return rgb_img, img, message

# cv2.destroyAllWindows()
# # ////////////////////////////////////////////////////END ////////////////////////////////////////////////////











##### FINAL SCRIPT WITH PLOT AND ALL ################
import cv2
import numpy as np

def detect_and_draw_circles(image_path, scale_factor=0.5):
    # Load the image
    img = cv2.imread(image_path)
    original_shape = img.shape[:2]
    
    # Resize image for faster processing
    width = int(original_shape[1] * scale_factor)
    height = int(original_shape[0] * scale_factor)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Apply mean filter
    kernelMean = np.ones((3,3), np.float32) / 9
    imgMean = cv2.filter2D(gray, -1, kernelMean)


    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(imgMean, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imgMean, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Apply thresholding to obtain binary edges
    threshold = 67 # 67 Adjusted threshold for faster processing
    edges = np.uint8(gradient_magnitude > threshold) * 255

    # Convert edges to binary image
    ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    
    # Define parameter values
    param1 = 200#100 178 Upper threshold for the edge detector
    param2 = 13# 25 Threshold for circle detection
    dp = 1        # Inverse ratio of the accumulator resolution to the image resolution
    minDist = img.shape[0] // 4  # Minimum distance between circle centers

    # Detect circles
    circles = cv2.HoughCircles(binary_edges,
                               cv2.HOUGH_GRADIENT,
                               dp=dp,
                               minDist=minDist,
                               param1=param1,
                               param2=param2,
                               minRadius=32,
                               maxRadius=35)

    if circles is not None:
        # Convert circles to integer
        circles = np.uint16(np.around(circles))
        # Sort circles by radius in descending order and select the largest one
        circles_sorted = sorted(circles[0, :1], key=lambda x: x[2], reverse=True)
        best_circle = circles_sorted[0]  # Select the largest circle
        center = (int(best_circle[0] * (1 / scale_factor)), int(best_circle[1] * (1 / scale_factor)))
        radius = int(best_circle[2] * (1 / scale_factor))
        # Draw the best circle on the original image
        cv2.circle(img, center, radius, (0, 255, 0), 2)
        messages = ["inside"]
    else:
        messages = ["Overhanging"]
    
    return img, messages
