# from picamera2 import Picamera2
# import cv2
# import os

# # Create directories if they don't exist
# left_dir = 'images/stereoLeft/'
# right_dir = 'images/stereoRight/'
# os.makedirs(left_dir, exist_ok=True)
# os.makedirs(right_dir, exist_ok=True)

# camera = Picamera2(0)
# camera2 = Picamera2(1)

# num = 0

# while True:
#     camera.start()
#     camera2.start()

#     img = camera.capture_array()
#     img2 = camera2.capture_array()

#     # Convert images to RGB
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#     k = cv2.waitKey(5)

#     if k == 27:
#         break
#     elif k == ord('s'):  # wait for 's' key to save and exit
#         cv2.imwrite(left_dir + 'imageL' + str(num) + '.png', img_rgb)
#         cv2.imwrite(right_dir + 'imageR' + str(num) + '.png', img2_rgb)
#         print("images saved!")
#         num += 1

#     cv2.imshow('Img 1', img_rgb)
#     cv2.imshow('Img 2', img2_rgb)

# # Release resources
# camera.stop()
# camera2.stop()
# cv2.destroyAllWindows()



#############Version2###############

import cv2
import time
import os
from picamera2 import Picamera2

# Create the capture directory if it doesn't exist
if not os.path.exists("capture"):
    os.makedirs("capture")

# Initialize the camera
picam2 = Picamera2()

# Set the resolution
w, h = 3280, 2464
camera_config = picam2.create_still_configuration(main={"size": (w, h)})
picam2.configure(camera_config)

# Start the camera
picam2.start()

time.sleep(3)

i = 0
while True:
    # Capture a frame
    frame_0 = picam2.capture_array()

    # Display the frame
    cv2.imshow("frame 0", cv2.resize(frame_0, (0,0), fx=0.25, fy=0.25))

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

    if key == ord("s"):
        print("saved image %d" % i)
        cv2.imwrite("capture/cam_0_%d.jpg" % i, frame_0)
        i += 1

    if i == 20:
        break

# Stop the camera and close the window
cv2.destroyAllWindows()
picam2.stop()
