import numpy as np
import cv2
from matplotlib import pyplot as plt

left_img = cv2.imread('left.png', 0)
right_img = cv2.imread('right.png', 0)

# Pre-filtering - Apply a Gaussian blur
left_img = cv2.GaussianBlur(left_img, (5, 5), 0)
right_img = cv2.GaussianBlur(right_img, (5, 5), 0)

num_disparities = 6 * 16  # Needs to be divisible by 16, increase for more detail
block_size = 15  # A larger block size encourages smoother, though less detailed disparities

# Adjust the P1 and P2 parameters for smoothness
P1 = 8 * block_size**2
P2 = 32 * block_size**2

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=P1,
    P2=P2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute the disparity map
disparity = stereo.compute(left_img, right_img)

# Normalize the disparity map for display
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the disparity map
plt.imshow(disparity_normalized, 'gray')
plt.title('Disparity Map')
plt.colorbar()  # Adding a colorbar to help interpret the values
plt.show()
