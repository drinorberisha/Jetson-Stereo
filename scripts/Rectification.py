import cv2
import numpy as np

# Load the previously saved calibration data
print("Loading calibration and stereo data...")
data = np.load('../data/calibration_data_right.npz')
mtx = data['mtx']
dist = data['dist']
stereo_calib_data = np.load('stereo_calibration_data.npz')

# Load stereo images
print("Loading stereo images for rectification...")
imgL = cv2.imread('../images/data/img/leftcamera/Im_L_5.png')
imgR = cv2.imread('../images/data/img/rightcamera/Im_R_5.png')

# Convert to grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Compute the maps for remapping the images
print("Computing rectification maps...")
left_map_x, left_map_y = cv2.initUndistortRectifyMap(mtx, dist, stereo_calib_data['R1'], stereo_calib_data['P1'], grayL.shape[::-1], cv2.CV_16SC2)
right_map_x, right_map_y = cv2.initUndistortRectifyMap(mtx, dist, stereo_calib_data['R2'], stereo_calib_data['P2'], grayR.shape[::-1], cv2.CV_16SC2)

# Remap the images
rectified_left = cv2.remap(grayL, left_map_x, left_map_y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(grayR, right_map_x, right_map_y, cv2.INTER_LINEAR)

cv2.imwrite(f'../results/rectified_images/rectified_left.jpg', rectified_left)
cv2.imwrite(f'../results/rectified_images/rectified_right.jpg', rectified_right)
print("Rectified images saved in '/results/rectified_images'.")

# Display or save the rectified images
print("Displaying rectified images...")
cv2.imshow('Rectified Left', rectified_left)
cv2.imshow('Rectified Right', rectified_right)
cv2.waitKey(0)
cv2.destroyAllWindows()
