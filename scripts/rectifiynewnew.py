import cv2
import numpy as np
import glob
import os

def load_calibration_parameters(file_path):
    data = np.load(file_path)
    return {
        "mtx_left": data['mtx_left'],
        "dist_left": data['dist_left'],
        "mtx_right": data['mtx_right'],
        "dist_right": data['dist_right'],
        "R1": data['R1'],
        "R2": data['R2'],
        "P1": data['P1'],
        "P2": data['P2'],
        "valid_roi1": data['valid_roi1'],
        "valid_roi2": data['valid_roi2'],
        "img_size": tuple(data['img_size']),
        "R": data['R'],
        "T": data['T']
    }

def rectify_images(img1, img2, params):
    mtx_left = params['mtx_left']
    dist_left = params['dist_left']
    mtx_right = params['mtx_right']
    dist_right = params['dist_right']
    R1 = params['R1']
    R2 = params['R2']
    P1 = params['P1']
    P2 = params['P2']
    valid_roi1 = params['valid_roi1']
    valid_roi2 = params['valid_roi2']
    img_size = params['img_size']

    rect_left = cv2.undistort(img1, mtx_left, dist_left, None, P1)
    rect_right = cv2.undistort(img2, mtx_right, dist_right, None, P2)

    if rect_left is None or rect_right is None:
        print("Rectification failed for images.")
        return None, None

    print(f"Valid ROI1: {valid_roi1}")
    print(f"Valid ROI2: {valid_roi2}")

    x1, y1, w1, h1 = valid_roi1
    x2, y2, w2, h2 = valid_roi2

    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        print("Invalid cropping dimensions for images.")
        return None, None

    rect_left = rect_left[y1:y1+h1, x1:x1+w1]
    rect_right = rect_right[y2:y2+h2, x2:x2+w2]

    return rect_left, rect_right

# Load calibration parameters
calibration_params = load_calibration_parameters('../data/stereo_calibration_data.npz')

# Directory containing the valid image pairs
valid_pairs_dir = '../images/valid_pairs'
rectified_dir = '../images/rectified_frames'
if not os.path.exists(rectified_dir):
    os.makedirs(rectified_dir)

# Load valid image pairs
left_image_files = sorted(glob.glob(os.path.join(valid_pairs_dir, 'valid_left_*.jpg')))
right_image_files = sorted(glob.glob(os.path.join(valid_pairs_dir, 'valid_right_*.jpg')))

if not left_image_files or not right_image_files or len(left_image_files) != len(right_image_files):
    print("No valid image pairs found in the directory.")
    exit()

for left_file, right_file in zip(left_image_files, right_image_files):
    left_frame = cv2.imread(left_file)
    right_frame = cv2.imread(right_file)

    if left_frame is None or right_frame is None:
        print(f"Failed to load image pair {left_file} and {right_file}")
        continue

    rect_left, rect_right = rectify_images(left_frame, right_frame, calibration_params)

    if rect_left is None or rect_right is None:
        continue

    basename_left = os.path.basename(left_file)
    basename_right = os.path.basename(right_file)

    left_rectified_path = os.path.join(rectified_dir, "rectified_" + basename_left)
    right_rectified_path = os.path.join(rectified_dir, "rectified_" + basename_right)

    cv2.imwrite(left_rectified_path, rect_left)
    cv2.imwrite(right_rectified_path, rect_right)

    combined_rectified = np.hstack((rect_left, rect_right))
    cv2.imshow("Rectified Images", combined_rectified)

    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
