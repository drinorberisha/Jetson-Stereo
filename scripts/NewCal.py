import numpy as np
import cv2
import glob
import os

def calibrate_camera(images, camera_name):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    gray = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            print(f"Detected corners in {fname}")
        else:
            print(f"Failed to detect corners in {fname}")
    cv2.destroyAllWindows()
    if objpoints and imgpoints:
        print(f"{camera_name} calibration process completed, saving data...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(f"{camera_name} Matrix: {mtx}")
        print(f"{camera_name} Distortion Coefficients: {dist}")

        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)
        print(f"{camera_name} total reprojection error: {mean_error}")

        return mtx, dist, rvecs, tvecs
    else:
        print(f"No valid images with detectable corners found for {camera_name} calibration.")
        return None

def main():
    os.makedirs('data', exist_ok=True)
    print("Starting calibration process...")
    left_images = glob.glob('../images/data/imgs/leftcamera/*.png')
    right_images = glob.glob('../images/data/imgs/rightcamera/*.png')
    left_camera_data = calibrate_camera(left_images, "Left Camera")
    right_camera_data = calibrate_camera(right_images, "Right Camera")

    if left_camera_data:
        np.savez('data/calibration_data_left.npz', mtx=left_camera_data[0], dist=left_camera_data[1], rvecs=left_camera_data[2], tvecs=left_camera_data[3])
        print("Left camera calibration data saved.")
    else:
        print("Failed to calibrate left camera.")

    if right_camera_data:
        np.savez('../data/calibration_data_right.npz', mtx=right_camera_data[0], dist=right_camera_data[1], rvecs=right_camera_data[2], tvecs=right_camera_data[3])
        print("Right camera calibration data saved.")
    else:
        print("Failed to calibrate right camera.")

if __name__ == "__main__":
    main()


# the values in the matrix [mtx] are fx, fy , cx , cy. fx and fy are the focal lengths of the camera expressed in pixel units.
# These values describe how much the camera magnifies the scene in terms of pixels per unit distance in the x and y directions, respectively.
# cx​ and cy​: These are the coordinates of the optical center (also known as the principal point) of the camera, expressed in pixel coordinates. 
# For your matrix, cx=572.83382086 and cy​=324.73669189.
# This point is ideally the center of the image sensor but can vary based on manufacturing variances or sensor alignme