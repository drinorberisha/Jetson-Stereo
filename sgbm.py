import cv2
import numpy as np

# Load the stereo images
left_img = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)
blockSize=5

# Create StereoSGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Adjust accordingly
    blockSize=blockSize,
    P1=8 * 3 * blockSize**2,
    P2=32 * 3 * blockSize**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute the disparity map
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

# Normalize and display the disparity map for visualization
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.imshow('Disparity', disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()