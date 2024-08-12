
# #####timer####
# from picamera2 import Picamera2
# import cv2
# import numpy as np
# import tempfile
# import time


# # Import your circle detection function
# from Perimiterdetect import detect_and_draw_circles
# from UndisortR import undistort_image


# def start_stream(picam2):
#     print("Starting stream...")
    
#     # Define the desired resolution
#     reduced_resolution = (800, 600)  # Set the desired resolution here

#     # Configure the camera for reduced resolution
#     camera_config = picam2.create_still_configuration(main={"size": reduced_resolution})
#     picam2.configure(camera_config)

#     picam2.start()
    
#     while True:
#         # Capture an image array
#         image = picam2.capture_array()
#         print("Captured image...")

#         # Convert the image to RGB format
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Save the RGB image array as a temporary file
#         try:
#             with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
#                 temp_file.write(cv2.imencode('.jpg', image_rgb)[1])
#             print("Temporary file created:", temp_file.name)
#         except Exception as e:
#             print("Error creating temporary file:", e)
#             break

#         # Call your circle detection function on the temporary file
#         try:
#             original_img, messages = detect_and_draw_circles(temp_file.name)
#             print("Circle detection completed.")
#         except Exception as e:
#             print("Error in circle detection:", e)
#             break

#         # Draw messages on the image
#         for message in messages:
#             cv2.putText(original_img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Display the processed images
#         cv2.imshow('Original Image', original_img)
#         print("Displaying processed image...")

#         # Wait for 0.5 seconds before capturing the next image
#         time.sleep(0.5)

#         # Exit the loop if 'm' is pressed
#         key = cv2.waitKey(1)
#         if key == ord('m'):
#             break

#     # Stop the camera before exiting
#     picam2.stop()
#     print("Stopped camera.")

#     # Close OpenCV windows
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Initialize Picamera2
#     picam2 = Picamera2()

#     # Configure the camera for reduced resolution
#     reduced_resolution = (800, 600)  # Set the desired resolution here
#     camera_config = picam2.create_still_configuration(main={"size": reduced_resolution})
#     picam2.configure(camera_config)

#     print("Camera configured with reduced resolution:", reduced_resolution)

#     # Start the stream
#     start_stream(picam2)



##############version2 with undisortion ############################
from picamera2 import Picamera2
import cv2
import numpy as np
import tempfile
import time

# Import your circle detection function
from Perimiterdetect import detect_and_draw_circles
from UndisortR import undistort_image

def start_stream(picam2):
    print("Starting stream...")
    
    # Define the desired resolution
    reduced_resolution = (800, 600)  # Set the desired resolution here

    # Configure the camera for reduced resolution
    camera_config = picam2.create_still_configuration(main={"size": reduced_resolution})
    picam2.configure(camera_config)

    picam2.start()
    
    while True:
        # Capture an image array
        image = picam2.capture_array()
        print("Captured image...")

        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Undistort the image
        undistorted_img = undistort_image(image_rgb)

        # Save the undistorted image as a temporary file
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(cv2.imencode('.jpg', undistorted_img)[1])
            print("Temporary file created:", temp_file.name)
        except Exception as e:
            print("Error creating temporary file:", e)
            break

        # Call your circle detection function on the temporary file
        try:
            original_img, messages = detect_and_draw_circles(temp_file.name)
            print("Circle detection completed.")
        except Exception as e:
            print("Error in circle detection:", e)
            break

        # Draw messages on the image
        for message in messages:
            cv2.putText(original_img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the processed images
        cv2.imshow('Original Image', original_img)
        print("Displaying processed image...")

        # Wait for 0.5 seconds before capturing the next image
        time.sleep(0.5)

        # Exit the loop if 'm' is pressed
        key = cv2.waitKey(1)
        if key == ord('m'):
            break

    # Stop the camera before exiting
    picam2.stop()
    print("Stopped camera.")

    # Close OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize Picamera2
    picam2 = Picamera2()

    # Configure the camera for reduced resolution
    reduced_resolution = (800, 600)  # Set the desired resolution here
    camera_config = picam2.create_still_configuration(main={"size": reduced_resolution})
    picam2.configure(camera_config)

    print("Camera configured with reduced resolution:", reduced_resolution)

    # Start the stream
    start_stream(picam2)
