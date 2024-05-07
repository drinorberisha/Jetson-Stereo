import cv2
import numpy as np

# Load the stereo images
left_img = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# Preprocessing with bilateral filter to reduce noise and preserve edges
left_img_filtered = cv2.bilateralFilter(left_img, 9, 75, 75)
right_img_filtered = cv2.bilateralFilter(right_img, 9, 75, 75)

blockSize = 5

# Create StereoSGBM object with optimized parameters
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Adjust accordingly
    blockSize=blockSize,
    P1=8 * 3 * blockSize**2,
    P2=32 * 3 * blockSize**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,  # Slightly adjusted for better matching
    speckleWindowSize=50,  # Adjusted based on trial and error
    speckleRange=16,  # Adjusted for underwater conditions
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute the disparity map using filtered images
disparity = stereo.compute(left_img_filtered, right_img_filtered).astype(np.float32) / 16.0

# Post-processing with Weighted Least Squares Filter to refine the disparity map
lmbda = 80000
sigma = 1.2

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

disparity_filtered = wls_filter.filter(disparity, left_img, None, right_img)

# Replace NaN or INF values and ensure the values are within the 0-255 range
disparity_filtered_safe = np.nan_to_num(disparity_filtered, nan=0.0, posinf=255, neginf=0)
disparity_filtered_clipped = np.clip(disparity_filtered_safe, 0, 255)

# Convert to uint8
disparity_filtered_uint8 = np.uint8(disparity_filtered_clipped)

# Normalize for visualization
disparity_filtered_normalized = cv2.normalize(disparity_filtered_uint8, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('Refined Disparity', disparity_filtered_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
