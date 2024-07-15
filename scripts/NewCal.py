import numpy as np
import cv2
import glob
import os

def split_stereo_image(stereo_image):
    height, width = stereo_image.shape[:2]
    midpoint = width // 2
    left_img = stereo_image[:, :midpoint]
    right_img = stereo_image[:, midpoint:]
    return left_img, right_img

def calibrate_single_camera(images, square_size, side):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    for fname in images:
        stereo_img = cv2.imread(fname)
        if stereo_img is None:
            print(f"Failed to load {fname}")
            continue

        left_img, right_img = split_stereo_image(stereo_img)
        img = left_img if side == 'left' else right_img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow(f'{side.capitalize()} Image', img)
            cv2.waitKey(500)
            print(f"Detected corners in {fname} on {side} side")
        else:
            print(f"Failed to detect corners in {fname} on {side} side")
    cv2.destroyAllWindows()

    if objpoints and imgpoints:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist, rvecs, tvecs
    else:
        print(f"No valid images with detectable corners found for {side} camera calibration.")
        return None

def calibrate_stereo_camera(images, square_size):
    left_calib = calibrate_single_camera(images, square_size, 'left')
    right_calib = calibrate_single_camera(images, square_size, 'right')

    if left_calib and right_calib:
        left_mtx, left_dist, left_rvecs, left_tvecs = left_calib
        right_mtx, right_dist, right_rvecs, right_tvecs = right_calib

        objpoints = []
        left_imgpoints = []
        right_imgpoints = []

        for fname in images:
            stereo_img = cv2.imread(fname)
            if stereo_img is None:
                print(f"Failed to load {fname}")
                continue

            left_img, right_img = split_stereo_image(stereo_img)
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(left_gray, (7, 6), None)
            ret_right, corners_right = cv2.findChessboardCorners(right_gray, (7, 6), None)

            if ret_left and ret_right:
                objp = np.zeros((6*7, 3), np.float32)
                objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * square_size

                objpoints.append(objp)
                corners2_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
                corners2_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
                left_imgpoints.append(corners2_left)
                right_imgpoints.append(corners2_right)
            else:
                print(f"Failed to detect corners in {fname} on both sides")

        if objpoints and left_imgpoints and right_imgpoints:
            _, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                objpoints, left_imgpoints, right_imgpoints,
                left_mtx, left_dist, right_mtx, right_dist,
                left_gray.shape[::-1], criteria=criteria
            )
            return left_mtx, left_dist, right_mtx, right_dist, R, T, E, F
        else:
            print("Failed to collect valid stereo pairs for calibration.")
            return None
    else:
        print("Single camera calibration failed.")
        return None

def main():
    os.makedirs('data', exist_ok=True)
    print("Starting calibration process...")

    square_size = 30  # Define the square size of the chessboard (e.g., 30 mm)

    stereo_images = glob.glob('../images/saved_frames/*.jpg')

    stereo_calib_data = calibrate_stereo_camera(stereo_images, square_size)

    if stereo_calib_data:
        left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = stereo_calib_data
        np.savez('../data/stereo_calibration_data.npz', left_mtx=left_mtx, left_dist=left_dist, right_mtx=right_mtx, right_dist=right_dist, R=R, T=T, E=E, F=F)
        print("Stereo calibration data saved.")
        
        data = np.load('../data/stereo_calibration_data.npz')
        print(f"Loaded stereo calibration data:\nleft_mtx: {data['left_mtx']}\nleft_dist: {data['left_dist']}\nright_mtx: {data['right_mtx']}\nright_dist: {data['right_dist']}\nR: {data['R']}\nT: {data['T']}\nE: {data['E']}\nF: {data['F']}")
    else:
        print("Failed to calibrate stereo camera.")

if __name__ == "__main__":
    main()
