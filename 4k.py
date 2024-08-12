# # ////////////////////////////////////////////////////////////FIRST SCRIPT /////////////////////////////////////////////////////////////////////////////////////////////////////////////

# from picamera2 import Picamera2
# import cv2
# import numpy as np
# import tempfile

# # Import your circle detection function
# from Perimiterdetect import detect_and_draw_circles

# def start_stream(picam2_1, picam2_2):
#     picam2_1.start()
#     picam2_2.start()
#     while True:
#         # Capture image arrays from both cameras
#         image1 = picam2_1.capture_array()
#         image2 = picam2_2.capture_array()

#         # Process the first camera's image
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file1:
#             temp_file1.write(cv2.imencode('.jpg', image1)[1])
#         img, original_img1, messages1 = detect_and_draw_circles(temp_file1.name)
#         for message in messages1:
#             cv2.putText(original_img1, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # # Process the second camera's image
#         # with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file2:
#         #     temp_file2.write(cv2.imencode('.jpg', image2)[1])
#         # img, original_img2, messages2 = detect_and_draw_circles(temp_file2.name)
#         # for message in messages2:
#         #     cv2.putText(original_img2, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # # Check if both messages contain message1
#         # if messages1[0] and messages2[0]:
#         #     depth_map = findobject(img1, img2)
#         #     # Display the depth map or process it further
#         #     cv2.imshow('Depth Map', depth_map)


#         # Display the processed images from both cameras
#         cv2.imshow('Original Image 1', original_img1)
#         cv2.imshow('Original Image 2', original_img2)

#         # Exit the loop if 'm' is pressed
#         key = cv2.waitKey(1)
#         if key == ord('m'):
#             break

#     # Stop the cameras before exiting
#     picam2_1.stop()
#     picam2_2.stop()

#     # Close OpenCV windows
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Initialize Picamera2 objects for both cameras
#     picam2_1 = Picamera2(0)
#     picam2_2 = Picamera2(1)

#     # Start the stream
#     start_stream(picam2_1, picam2_2)

# ////////////////////////////////////////////////////THE END //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# import cv2
# import numpy as np
# import tempfile
# from picamera2 import Picamera2
# from Perimiterdetect import detect_and_draw_circles
# from dephtperi import findobject

# def start_stream(picam2_1, picam2_2):
#     picam2_1.start()
#     picam2_2.start()
#     while True:
#         # Capture image arrays from both cameras
#         image1 = picam2_1.capture_array()
#         image2 = picam2_2.capture_array()

#         # Process the first camera's image
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file1:
#             temp_file1.write(cv2.imencode('.jpg', image1)[1])
#         img1, original_img1, message1 = detect_and_draw_circles(temp_file1.name)
#         cv2.putText(original_img1, message1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Process the second camera's image
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file2:
#             temp_file2.write(cv2.imencode('.jpg', image2)[1])
#         img2, original_img2, message2 = detect_and_draw_circles(temp_file2.name)
#         cv2.putText(original_img2, message2, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Check if the first messages in both lists are "Object within workspace."
#         if message1 == "Object within workspace." and message2 == "Object within workspace.":
#             depthimg, binary = findobject(img1, img2)
#             cv2.imshow('Depth Map', depthimg)
#             cv2.imshow('Depth Map', binary)
#             # Display the depth map or process it further
       


#         # Display the processed images from both cameras
#         # cv2.imshow('Original Image 1', original_img1) , the i
#         cv2.imshow('Original Image 2', img2)
        

#         # Exit the loop if 'm' is pressed
#         key = cv2.waitKey(1)
#         if key == ord('m'):
#             b # # Process the second camera's image
        # with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file2:
        #     temp_file2.write(cv2.imencode('.jpg', image2)[1])
        # img, original_img2, messages2 = detect_and_draw_circles(temp_file2.name)
        # for message in messages2:
        #     cv2.putText(original_img2, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)reak

#     # Stop the cameras before exiting
#     picam2_1.stop()
#     picam2_2.stop()

#     # Close OpenCV windows
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Initialize Picamera2 objects for both cameras
#     picam2_1 = Picamera2(0)
#     picam2_2 = Picamera2(1)

#     # Start the stream
#     start_stream(picam2_1, picam2_2)







 
# ################################################################ 4K with ethernet Communication ################################################

# from picamera2 import Picamera2
# import cv2
# import numpy as np
# import tempfile
# import time

# # Import your circle detection function
# from Perimiterdetect import detect_and_draw_circles

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




# ###################### 4k ethernet full script ###################################
# from picamera2 import Picamera2
# import cv2
# import numpy as np
# import tempfile
# import socket

# from Perimiterdetect import detect_and_draw_circles

# def start_stream(picam2, connection):
#     picam2.start()
#     while True:
#         try:
#             # Capture an image array
#             image = picam2.capture_array()

#             # Save the image array as a temporary file
#             with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
#                 temp_file.write(cv2.imencode('.jpg', image)[1])

#             # Call your circle detection function on the temporary file
#             original_img, messages = detect_and_draw_circles(temp_file.name)

#             # Draw messages on the image
#             for message in messages:
#                 cv2.putText(original_img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#                 connection.sendall(message.encode('ascii'))  # Send message to the client

#             # Display the processed images
#             # cv2.imshow('Original Image', original_img)

#             # Exit the loop if 'm' is pressed
#             key = cv2.waitKey(1)
#             if key == ord('m'):
#                 break
#         except Exception as e:
#             print("Error during streaming:", e)
#             break

#     # Stop the camera before exiting
#     picam2.stop()
#     # Close OpenCV windows
#     cv2.destroyAllWindows()

# def setup_server(ip_address, port):
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     try:
#         server.bind((ip_address, port))
#         server.listen(1)
#         print("Server listening on", ip_address, ":", port)

#         connection, address = server.accept()
#         print("Connection from", address)
#         return connection
#     except Exception as e:
#         print("Error setting up server:", e)
#         server.close()
#         return None

# if __name__ == "__main__":
#     # IP address and port to listen on
#     ip_address = "192.168.2.10"  # Raspberry Pi's IP address
#     port = 65431

#     # Setup server and accept connection
#     connection = setup_server(ip_address, port)
#     if connection:
#         picam2 = Picamera2()

#         # Configure the camera for full resolution
#         camera_config = picam2.create_still_configuration(main={"size": picam2.sensor_resolution})
#         picam2.configure(camera_config)

#         try:
#             # Start the stream
#             start_stream(picam2, connection)
#         finally:
#             connection.close()
#             print("Connection closed")




#####new one ####

from picamera2 import Picamera2
import cv2
import numpy as np
import tempfile
import time
import socket

# Import your circle detection function
from Perimiterdetect import detect_and_draw_circles

def start_stream(picam2, connection):
    print("Starting stream...")

    # Define the desired resolution
    reduced_resolution = (800, 600)  # Set the desired resolution here

    # Configure the camera for reduced resolution
    camera_config = picam2.create_still_configuration(main={"size": reduced_resolution})
    picam2.configure(camera_config)

    picam2.start()

    while True:
        try:
            # Capture an image array
            image = picam2.capture_array()
            print("Captured image...")

            # Convert the image to RGB format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save the RGB image array as a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(cv2.imencode('.jpg', image_rgb)[1])
            print("Temporary file created:", temp_file.name)

            # Call your circle detection function on the temporary file
            original_img, messages = detect_and_draw_circles(temp_file.name)
            print("Circle detection completed.")

            # Draw messages on the image and send them over the connection
            for message in messages:
                cv2.putText(original_img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                connection.sendall(message.encode('ascii'))  # Send message to the client

            # Display the processed images
            # cv2.imshow('Original Image', original_img)
            # print("Displaying processed image...")

            # Wait for 0.5 seconds before capturing the next image
            time.sleep(0.5)

            # Exit the loop if 'm' is pressed
            key = cv2.waitKey(1)
            if key == ord('m'):
                break

        except Exception as e:
            print("Error during streaming:", e)
            break

    # Stop the camera before exiting
    picam2.stop()
    print("Stopped camera.")

    # Close OpenCV windows
    cv2.destroyAllWindows()

def setup_server(ip_address, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((ip_address, port))
        server.listen(1)
        print("Server listening on", ip_address, ":", port)

        connection, address = server.accept()
        print("Connection from", address)
        return connection
    except Exception as e:
        print("Error setting up server:", e)
        server.close()
        return None

if __name__ == "__main__":
    # IP address and port to listen on
    ip_address = "192.168.2.10"  # Raspberry Pi's IP address
    port = 65431

    # Setup server and accept connection
    connection = setup_server(ip_address, port)
    if connection:
        picam2 = Picamera2()

        # Start the stream with serial communication
        try:
            start_stream(picam2, connection)
        finally:
            connection.close()
            print("Connection closed")

