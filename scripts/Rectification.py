import cv2
import numpy as np
import glob
import os

# Load calibration parameters
calibration_data = np.load('../data/stereo_calibration_data.npz')
mtx_left = calibration_data['mtx_left']
dist_left = calibration_data['dist_left']
mtx_right = calibration_data['mtx_right']
dist_right = calibration_data['dist_right']
R1 = calibration_data['R1']
R2 = calibration_data['R2']
P1 = calibration_data['P1']
P2 = calibration_data['P2']

# Directory containing the saved images
images_dir = 'images/saved_frames'
rectified_dir = 'images/rectified_frames'
if not os.path.exists(rectified_dir):
    os.makedirs(rectified_dir)

# Load images
image_files = sorted(glob.glob(os.path.join(images_dir, 'jetsonchessboard_*.jpg')))
if not image_files:
    print("No images found in the directory.")
    exit()

for image_file in image_files:
    image = cv2.imread(image_file)
    height, width = image.shape[:2]
    midpoint = width // 2

    left_frame = image[:, :midpoint]
    right_frame = image[:, midpoint:]

    # Apply rectification
    img_size = (left_frame.shape[1], left_frame.shape[0])
    map_left1, map_left2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, img_size, cv2.CV_16SC2)
    map_right1, map_right2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, img_size, cv2.CV_16SC2)

    rect_left = cv2.remap(left_frame, map_left1, map_left2, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_frame, map_right1, map_right2, cv2.INTER_LINEAR)

    # Save rectified images
    basename = os.path.basename(image_file)
    left_rectified_path = os.path.join(rectified_dir, f"rectified_left_{basename}")
    right_rectified_path = os.path.join(rectified_dir, f"rectified_right_{basename}")

    cv2.imwrite(left_rectified_path, rect_left)
    cv2.imwrite(right_rectified_path, rect_right)

    # Display the rectified images
    images = np.hstack((rect_left, rect_right))
    cv2.imshow("Rectified Images", images)

    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
