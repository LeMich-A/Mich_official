import numpy as np
import cv2
import glob

# Parameter
CHESSBOARD_SIZE = (6, 9)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real-world space
imgpoints = []  # 2d points in image plane.

# Load images
imagesLeft = sorted(glob.glob('images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('imageRight/*.jpg'))

# Check if images are loaded
if len(imagesLeft) == 0:
    print("No left images found in the specified directory. Exiting...")
elif len(imagesRight) == 0:
    print("No right images found in the specified directory. Exiting...")
else:
    for i, imgRight in enumerate(imagesRight):
        print(f"Processing image {i+1}/{len(imagesRight)}: {imgRight}")

        imgR = cv2.imread(imgRight)
        
        if imgR is None:
            print(f"Failed to load image: {imgRight}. Skipping...")
            continue

        imgR = cv2.resize(imgR, (0, 0), fx=0.25, fy=0.25)
        print(f"Image shape after resizing: {imgR.shape}")
        
        gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        print(f"Finding chessboard corners in {imgRight}...")
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            print("Chessboard corners found!")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            print("Drawing corners on the image.")
            cv2.drawChessboardCorners(imgR, CHESSBOARD_SIZE, corners2, ret)
            cv2.imshow('Chessboard Corners', imgR)
            cv2.waitKey(500)  # Show the image for 500ms
        else:
            print(f"Chessboard corners not found in {imgRight}. Skipping...")

    cv2.destroyAllWindows()

    # Check if object points and image points have been populated
    if objpoints and imgpoints:
        print("Object points and image points have been successfully populated.")
        h, w = imgR.shape[:2]
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
