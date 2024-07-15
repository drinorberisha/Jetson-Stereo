import cv2
import numpy as np
import os
import glob

def find_chessboard_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(image, board_size, corners, ret)
        cv2.imshow('Corners', image)
        cv2.waitKey(500)
    return ret, corners

def stereo_calibrate(left_images, right_images, board_size, square_size):
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    img_size = None

    for idx, (left_img_folder, right_img_folder) in enumerate(zip(left_images, right_images)):
        left_img = cv2.imread(left_img_folder)
        right_img = cv2.imread(right_img_folder)

        if left_img is None or right_img is None:
            print(f"Skipping image pair {idx} due to read error.")
            continue

        if img_size is None:
            img_size = (left_img.shape[1], left_img.shape[0])

        ret_left, corners_left = find_chessboard_corners(left_img, board_size)
        ret_right, corners_right = find_chessboard_corners(right_img, board_size)

        if ret_left and ret_right:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
        else:
            print(f"Skipping image pair {idx} due to chessboard corner detection failure.")

    if len(imgpoints_left) == 0 or len(imgpoints_right) == 0:
        print("Error: No valid image points were found.")
        return None

    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, img_size, None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, img_size, None, None)

    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

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
    square_size = 30  # Adjust this value based on your chessboard square size (in mm)
    board_size = (7, 6)  # Number of inner corners in the chessboard pattern
    left_image_folder = '../images/data/imgs/leftcamera'  # Folder containing left camera images
    right_image_folder = '../images/data/imgs/rightcamera'  # Folder containing right camera images

    left_image_files = sorted(glob.glob(os.path.join(left_image_folder, '*.png')))
    right_image_files = sorted(glob.glob(os.path.join(right_image_folder, '*.png')))

    if len(left_image_files) != len(right_image_files):
        print("Error: The number of left and right images do not match.")
        return

    # Display images to verify they contain the chessboard pattern
    for idx, (left_img_path, right_img_path) in enumerate(zip(left_image_files, right_image_files)):
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        if left_img is None or right_img is None:
            print(f"Error: Failed to load image pair {idx}")
            continue
        combined = np.hstack((left_img, right_img))
        cv2.imshow(f"Image Pair {idx}", combined)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

    calibration_data = stereo_calibrate(left_image_files, right_image_files, board_size, square_size)
    if calibration_data is None:
        print("Stereo calibration failed.")
        return

    # Save calibration data to file
    np.savez('stereo_calibration_data.npz', **calibration_data)
    print("Stereo calibration data saved.")

    # Load calibration data from file and display rectified images
    data = np.load('stereo_calibration_data.npz')
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

    for idx, (left_img_path, right_img_path) in enumerate(zip(left_image_files, right_image_files)):
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        if left_img is None or right_img is None:
            continue

        map_left1, map_left2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, (left_img.shape[1], left_img.shape[0]), cv2.CV_16SC2)
        map_right1, map_right2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, (right_img.shape[1], right_img.shape[0]), cv2.CV_16SC2)

        rect_left = cv2.remap(left_img, map_left1, map_left2, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_img, map_right1, map_right2, cv2.INTER_LINEAR)

        combined_rectified = np.hstack((rect_left, rect_right))
        cv2.imshow('Rectified', combined_rectified)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
