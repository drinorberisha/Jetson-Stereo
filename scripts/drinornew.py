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
valid_roi1 = calibration_data['valid_roi1']
valid_roi2 = calibration_data['valid_roi2']
img_size = tuple(calibration_data['img_size'])

# Directory containing the valid image pairs
valid_pairs_dir = '../images/valid_pairs'
rectified_dir = '../images/rectified_frames'
if not os.path.exists(rectified_dir):
    os.makedirs(rectified_dir)

# Load valid image pairs
left_image_files = sorted(glob.glob(os.path.join(valid_pairs_dir, 'Im_L_*.png')))
right_image_files = sorted(glob.glob(os.path.join(valid_pairs_dir, 'Im_R_*.png')))

if not left_image_files or not right_image_files or len(left_image_files) != len(right_image_files):
    print("No valid image pairs found in the directory.")
    exit()

for left_file, right_file in zip(left_image_files, right_image_files):
    left_frame = cv2.imread(left_file)
    right_frame = cv2.imread(right_file)

    if left_frame is None or right_frame is None:
        print(f"Failed to load image pair {left_file} and {right_file}")
        continue

    # Apply rectification
    rect_left = cv2.undistort(left_frame, mtx_left, dist_left, None, P1)
    rect_right = cv2.undistort(right_frame, mtx_right, dist_right, None, P2)

    # Debugging: Check if rectification maps are applied correctly
    if rect_left is None or rect_right is None:
        print(f"Rectification failed for image pair {left_file} and {right_file}")
        continue

    # Debugging: Print valid ROI values
    print(f"Valid ROI1: {valid_roi1}")
    print(f"Valid ROI2: {valid_roi2}")

    # Crop the rectified images
    x1, y1, w1, h1 = valid_roi1
    x2, y2, w2, h2 = valid_roi2

    # Check if cropping dimensions are valid
    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        print(f"Invalid cropping dimensions for image pair {left_file} and {right_file}")
        continue

    rect_left = rect_left[y1:y1+h1, x1:x1+w1]
    rect_right = rect_right[y2:y2+h2, x2:x2+w2]

    # Check if the images are empty after cropping
    if rect_left.size == 0:
        print(f"Left rectified image is empty after cropping for image pair {left_file} and {right_file}")
        continue
    if rect_right.size == 0:
        print(f"Right rectified image is empty after cropping for image pair {left_file} and {right_file}")
        continue

    # Save rectified images
    left_rectified_path = os.path.join(rectified_dir, os.path.basename(left_file).replace('valid_left_', 'rectified_left_'))
    right_rectified_path = os.path.join(rectified_dir, os.path.basename(right_file).replace('valid_right_', 'rectified_right_'))

    cv2.imwrite(left_rectified_path, rect_left)
    cv2.imwrite(right_rectified_path, rect_right)

    # Display the rectified images
    combined_rectified = np.hstack((rect_left, rect_right))
    cv2.imshow("Rectified Images", combined_rectified)

    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
