import cv2
import numpy as np
import os
import glob

def split_stereo_image(stereo_image):
    if stereo_image is None or len(stereo_image) == 0:
        print("Error: Stereo image is empty or None")
        return None, None

    height, width = stereo_image.shape[:2]
    midpoint = width // 2
    left_img = stereo_image[:, :midpoint]
    right_img = stereo_image[:, midpoint:]
    return left_img, right_img

def find_chessboard_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        print("Chessboard corners detected.")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(image, board_size, corners, ret)
    else:
        print("Error: Chessboard corners not detected.")
    return ret, corners, image

def stereo_calibrate(stereo_images, board_size, square_size):
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    img_size = None

    for idx, stereo_img_path in enumerate(stereo_images):
        stereo_img = cv2.imread(stereo_img_path)
        left_img, right_img = split_stereo_image(stereo_img)

        if left_img is None or right_img is None:
            print("Skipping image pair {} due to split error.".format(idx))
            continue

        if img_size is None:
            img_size = (left_img.shape[1], left_img.shape[0])

        ret_left, corners_left, left_img_with_corners = find_chessboard_corners(left_img, board_size)
        ret_right, corners_right, right_img_with_corners = find_chessboard_corners(right_img, board_size)

        if ret_left and ret_right:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

            # Save images with detected corners
            cv2.imwrite('../images/detectedCorners/left_with_corners_{}.jpg'.format(idx), left_img_with_corners)
            cv2.imwrite('../images/detectedCorners/right_with_corners_{}.jpg'.format(idx), right_img_with_corners)
        else:
            print("Skipping image pair {} due to chessboard corner detection failure.".format(idx))

    if len(imgpoints_left) == 0 or len(imgpoints_right) == 0:
        print("Error: No valid image points were found.")
        return None

    # Calibrate the left camera
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, img_size, None, None)

    # Calibrate the right camera
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, img_size, None, None)

    # Perform stereo calibration
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    # Perform rectification
    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    return {
        "mtx_left": mtx_left, "dist_left": dist_left,
        "mtx_right": mtx_right, "dist_right": dist_right,
        "R": R, "T": T, "E": E, "F": F,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "valid_roi1": valid_roi1, "valid_roi2": valid_roi2
    }

def main():
    square_size = 30  # Size of the chessboard square in mm
    board_size = (7, 6)  # Number of inner corners in the chessboard pattern
    image_folder = 'images/saved_frames'  # Folder containing stereo images

    stereo_image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

    # Display images to verify they contain the chessboard pattern and are split correctly
    for idx, stereo_img_path in enumerate(stereo_image_files):
        stereo_img = cv2.imread(stereo_img_path)
        left_img, right_img = split_stereo_image(stereo_img)
        if left_img is None or right_img is None:
            print("Error: Failed to split image {}".format(stereo_img_path))
            continue
        combined = np.hstack((left_img, right_img))
        cv2.imshow("Image Pair {}".format(idx), combined)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

    calibration_data = stereo_calibrate(stereo_image_files, board_size, square_size)
    if calibration_data is None:
        print("Stereo calibration failed.")
        return

    # Save calibration data to file
    np.savez('data/stereo_calibration_data.npz', **calibration_data)
    print("Stereo calibration data saved.")

    # Load calibration data from file and display rectified images
    data = np.load('data/stereo_calibration_data.npz')
    mtx_left = data['mtx_left']
    dist_left = data['dist_left']
    mtx_right = data['mtx_right']
    dist_right = data['dist_right']
    R1 = data['R1']
    R2 = data['R2']
    P1 = data['P1']
    P2 = data['P2']
    valid_roi1 = data['valid_roi1']
    valid_roi2 = data['valid_roi2']

    for idx, stereo_img_path in enumerate(stereo_image_files):
        stereo_img = cv2.imread(stereo_img_path)
        left_img, right_img = split_stereo_image(stereo_img)
        if left_img is None or right_img is None:
            continue

        map_left1, map_left2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, (left_img.shape[1], left_img.shape[0]), cv2.CV_16SC2)
        map_right1, map_right2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, (right_img.shape[1], right_img.shape[0]), cv2.CV_16SC2)

        rect_left = cv2.remap(left_img, map_left1, map_left2, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_img, map_right1, map_right2, cv2.INTER_LINEAR)

        combined_rectified = np.hstack((rect_left, rect_right))
        cv2.imwrite('../images/output/rectified_{}.jpg'.format(idx), combined_rectified)
        cv2.imshow('Rectified', combined_rectified)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
