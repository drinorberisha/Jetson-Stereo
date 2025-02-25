import cv2
import numpy as np
import os
import glob

def find_chessboard_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        print("Chessboard corners detected.")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(image, board_size, corners, ret)
    else:
        print("Error: Chessboard corners not detected.")
    return ret, corners, image

def stereo_calibrate(left_images, right_images, board_size, square_size):
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    img_size = None
    
    valid_image_dir = '../images/valid_pairs'
    if not os.path.exists(valid_image_dir):
        os.makedirs(valid_image_dir)

    for left_img_path, right_img_path in zip(left_images, right_images):
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        if left_img is None or right_img is None:
            print(f"Skipping image pair due to error in reading images.")
            continue

        if img_size is None:
            img_size = (left_img.shape[1], left_img.shape[0])

        ret_left, corners_left, left_img_with_corners = find_chessboard_corners(left_img, board_size)
        ret_right, corners_right, right_img_with_corners = find_chessboard_corners(right_img, board_size)

        if ret_left and ret_right:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

            left_filename = os.path.basename(left_img_path)
            right_filename = os.path.basename(right_img_path)
            valid_left_path = os.path.join(valid_image_dir, left_filename)
            valid_right_path = os.path.join(valid_image_dir, right_filename)
            cv2.imwrite(valid_left_path, left_img)
            cv2.imwrite(valid_right_path, right_img)
            corners_filename = f'corners_{os.path.splitext(left_filename)[0]}.npz'
            np.savez(os.path.join(valid_image_dir, corners_filename), 
                     corners_left=corners_left, corners_right=corners_right)

        else:
            print(f"Skipping image pair {left_img_path} and {right_img_path} due to chessboard corner detection failure.")

    if len(imgpoints_left) == 0 or len(imgpoints_right) == 0:
        print("Error: No valid image points were found.")
        return None

    print("Calibrating left camera...")
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, img_size, None, None)
    print(f"Left Camera Matrix:\n{mtx_left}")
    print(f"Left Distortion Coefficients:\n{dist_left}")

    print("Calibrating right camera...")
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, img_size, None, None)
    print(f"Right Camera Matrix:\n{mtx_right}")
    print(f"Right Distortion Coefficients:\n{dist_right}")

    print("Performing stereo calibration...")
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"Rotation Matrix:\n{R}")
    print(f"Translation Vector:\n{T}")
    print(f"Essential Matrix:\n{E}")
    print(f"Fundamental Matrix:\n{F}")

    alphas = [0]  # Testing five different alphas
    best_alpha = 0
    best_valid_roi = (0, 0, 0, 0)

    for alpha in alphas:
        print(f"\nPerforming stereo rectification with alpha={alpha}...")
        R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
            mtx_left, dist_left,
            mtx_right, dist_right,
            img_size, R, T,
            alpha=alpha,
            newImageSize=(0, 0)
        )

        print(f"Left Rectification Matrix (alpha={alpha}):\n{R1}")
        print(f"Right Rectification Matrix (alpha={alpha}):\n{R2}")
        print(f"Left Projection Matrix (alpha={alpha}):\n{P1}")
        print(f"Right Projection Matrix (alpha={alpha}):\n{P2}")
        print(f"Disparity-to-depth Mapping Matrix (alpha={alpha}):\n{Q}")
        print(f"Valid ROI1 (alpha={alpha}): {valid_roi1}")
        print(f"Valid ROI2 (alpha={alpha}): {valid_roi2}")

        # Update best alpha if this one has a valid ROI
        if valid_roi2[2] > 0 and valid_roi2[3] > 0 and valid_roi2[2] * valid_roi2[3] > best_valid_roi[2] * best_valid_roi[3]:
            best_alpha = alpha
            best_valid_roi = valid_roi2

    print(f"\nBest alpha value for valid ROI2: {best_alpha}")
    print(f"Valid ROI2 for best alpha: {best_valid_roi}")

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_left[i], tvecs_left[i], mtx_left, dist_left)
        error = cv2.norm(imgpoints_left[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Total reprojection error (left): {}".format(mean_error / len(objpoints)))

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_right[i], tvecs_right[i], mtx_right, dist_right)
        error = cv2.norm(imgpoints_right[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Total reprojection error (right): {}".format(mean_error / len(objpoints)))

    return {
        "mtx_left": mtx_left, "dist_left": dist_left,
        "mtx_right": mtx_right, "dist_right": dist_right,
        "R": R, "T": T, "E": E, "F": F,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "valid_roi1": valid_roi1, "valid_roi2": valid_roi2,
        "img_size": img_size
    }

def main():
    square_size = 30  # Size of the chessboard square in mm
    board_size = (11, 7)  
    image_folder_left = '../images/imgs/leftcamera'
    image_folder_right = '../images/imgs/rightcamera'

    left_images = sorted(glob.glob(os.path.join(image_folder_left, '*.png')))
    right_images = sorted(glob.glob(os.path.join(image_folder_right, '*.png')))

    if len(left_images) != len(right_images):
        print("Mismatch in the number of left and right images.")
        return

    calibration_data = stereo_calibrate(left_images, right_images, board_size, square_size)
    if calibration_data is None:
        print("Stereo calibration failed.")
        return

    np.savez('../data/stereo_calibration_data.npz', **calibration_data)
    print("Stereo calibration data saved.")

if __name__ == "__main__":
    main()
