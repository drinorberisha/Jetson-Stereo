import numpy as np
import cv2 as cv
import glob
import os

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Load images from both left and right camera folders
left_images = glob.glob('../images/data/imgs/leftcamera/*.png')
right_images = glob.glob('../images/data/imgs/rightcamera/*.png')

def process_images(images, camera_name):
    print(f"Processing images from {camera_name}...")
    for fname in images:
        print(f"Loading image {fname}...")
        img = cv.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            print(f"Chessboard corners found in {fname}")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
        else:
            print(f"No chessboard corners found in {fname}")

# Process both sets of images
process_images(left_images, "Left Camera")
process_images(right_images, "Right Camera")

cv.destroyAllWindows()
