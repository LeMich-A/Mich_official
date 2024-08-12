import numpy as np
import cv2
import os

# Parameter
CHESSBOARD_SIZE = (7, 9)

# Improved termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

# Prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Directory to read images from
capture_dir = "capture/"
if not os.path.exists(capture_dir):
    print(f"Directory '{capture_dir}' does not exist. Exiting...")
else:
    files = os.listdir(capture_dir)
    if not files:
        print(f"No images found in directory '{capture_dir}'. Exiting...")
    else:
        for filename in files:
            filepath = os.path.join(capture_dir, filename)
            print(f"Reading file: {filepath}")
            img = cv2.imread(filepath)
            
            if img is None:
                print(f"Failed to load image: {filename}. Skipping...")
                continue

            print(f"Original image shape: {img.shape}")
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            print(f"Image shape after resizing: {img.shape}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            print(f"Finding chessboard corners in {filename}...")
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            print(f"Chessboard corners found: {ret}")

            # If found, add object points, image points (after refining them)
            if ret:
                print("Chessboard corners detected!")
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                print("Drawing corners on the image.")
                cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
                cv2.imshow('Chessboard Corners', img)
                cv2.waitKey(500)  # Show the image for 500ms
            else:
                print(f"Chessboard corners not found in {filename}. Skipping...")

    cv2.destroyAllWindows()

    # Check if object points and image points have been populated
    if objpoints and imgpoints:
        print("Object points and image points have been successfully populated.")
        h, w = img.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print("Calibration successful!")
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist)
            print("Rotation vectors:\n", rvecs)
            print("Translation vectors:\n", tvecs)
        else:
            print("Calibration failed. No valid calibration parameters obtained.")
    else:
        print("No chessboard corners were found in any image. Calibration was not performed.")
