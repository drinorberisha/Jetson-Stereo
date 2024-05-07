import numpy as np
import cv2
from matplotlib import pyplot as plt

left_img = cv2.imread('left.png', 0)
right_img = cv2.imread('right.png', 0)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,  
    blockSize=11, 
    P1=8 * 3 * 11**2, 
    P2=32 * 3 * 11**2, 
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute the disparity map
disparity = stereo.compute(left_img, right_img)

# Normalize the disparity map for display
disparity_normalized = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the disparity map
plt.imshow(disparity_normalized, 'gray')
plt.show()
