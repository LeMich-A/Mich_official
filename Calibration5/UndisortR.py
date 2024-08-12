import cv2
import numpy as np

# Camera matrix and distortion coefficients
mtx = np.array([[760.01150397, 0.00000000e+00, 296.86221931],
                [0.00000000e+00, 727.17612765, 332.52898345],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[0.50884003, -0.86338934, 0.00261162, -0.10853514, 0.83252921]])

def undistort_image(img):
    """
    Undistorts the input image using the camera matrix and distortion coefficients.

    Parameters:
    img (numpy.ndarray): The input image to be undistorted.

    Returns:
    numpy.ndarray: The undistorted image.
    """
    h, w = img.shape[:2]

    # Refine the camera matrix based on a free scaling parameter
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image based on the ROI
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst

