import cv2
import numpy as np

# Load the stereo images
left_img = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

left_img_filtered = cv2.bilateralFilter(left_img, 9, 75, 75)
right_img_filtered = cv2.bilateralFilter(right_img, 9, 75, 75)


# Create stereo block matching object with optimized parameters
stereo = cv2.StereoBM_create(numDisparities=16*2, blockSize=15)  # Adjusted numDisparities for finer granularity

# Compute the disparity map using filtered images
disparity = stereo.compute(left_img_filtered, right_img_filtered)

# Post-processing with WLS Filter to refine the disparity map
# Note: WLS is typically used with SGBM, but a simple normalization here can help visualize the BMA result better
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the refined disparity map
cv2.imshow('Refined Disparity Map', disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
