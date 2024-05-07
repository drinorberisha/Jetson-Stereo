import cv2
import numpy as np

# Load the stereo images
left_img = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# Create stereo block matching object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(left_img, right_img)

# Normalize the disparity map for display
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Display the disparity map
cv2.imshow('Disparity Map', np.uint8(disparity_normalized))
cv2.waitKey(0)
cv2.destroyAllWindows()
