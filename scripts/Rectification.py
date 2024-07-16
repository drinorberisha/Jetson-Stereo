import cv2
import numpy as np

# Load calibration parameters
calibration_data = np.load('stereo_calibration_data.npz')
mtx_left = calibration_data['mtx_left']
dist_left = calibration_data['dist_left']
mtx_right = calibration_data['mtx_right']
dist_right = calibration_data['dist_right']
R1 = calibration_data['R1']
R2 = calibration_data['R2']
P1 = calibration_data['P1']
P2 = calibration_data['P2']
Q = calibration_data['Q']

# Capture new images
left_image = cv2.imread('left_image.jpg')
right_image = cv2.imread('right_image.jpg')

# Get image size
img_size = (left_image.shape[1], left_image.shape[0])

# Compute rectification maps
map_left1, map_left2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, img_size, cv2.CV_16SC2)
map_right1, map_right2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, img_size, cv2.CV_16SC2)

# Apply rectification
rect_left = cv2.remap(left_image, map_left1, map_left2, cv2.INTER_LINEAR)
rect_right = cv2.remap(right_image, map_right1, map_right2, cv2.INTER_LINEAR)

# Save or display rectified images
cv2.imwrite('rectified_left.jpg', rect_left)
cv2.imwrite('rectified_right.jpg', rect_right)
cv2.imshow('Rectified Left', rect_left)
cv2.imshow('Rectified Right', rect_right)
cv2.waitKey(0)
cv2.destroyAllWindows()
