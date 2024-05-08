import numpy as np
import cv2
import glob
import os


def split_stereo_image(stereo_image):
    # Assuming the stereo image is a single image composed of two side-by-side images
    height, width = stereo_image.shape[:2]
    midpoint = width // 2
    left_img = stereo_image[:, :midpoint]
    right_img = stereo_image[:, midpoint:]
    return left_img, right_img


def calibrate_camera(images):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for fname in images:
        stereo_img = cv2.imread(fname)
        if stereo_img is None:
            print(f"Failed to load {fname}")
            continue

        left_img, right_img = split_stereo_image(stereo_img)  # Split the stereo image
        for img, side in zip([left_img, right_img], ['Left', 'Right']):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                cv2.imshow(f'{side} Image', img)
                cv2.waitKey(500)
                print(f"Detected corners in {fname} on {side} side")
            else:
                print(f"Failed to detect corners in {fname} on {side} side")
        cv2.destroyAllWindows()
    if objpoints and imgpoints:
        print(f"Calibration process completed, saving data...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(f" Matrix: {mtx}")
        print(f" Distortion Coefficients: {dist}")

        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)
        print(f" total reprojection error: {mean_error}")

        return mtx, dist, rvecs, tvecs
    else:
        print(f"No valid images with detectable corners found for {camera_name} calibration.")
        return None

def main():
    os.makedirs('data', exist_ok=True)
    print("Starting calibration process...")
    stereo_images = glob.glob('../images/saved_frames/*.jpg')  # Adjust path as necessary
    camera_data = calibrate_camera(stereo_images)

    if camera_data:
        np.savez('../data/calibration_data.npz', mtx=camera_data[0], dist=camera_data[1], rvecs=camera_data[2], tvecs=camera_data[3])
        print("Camera calibration data saved.")
    else:
        print("Failed to calibrate camera.")


if __name__ == "__main__":
    main()


# the values in the matrix [mtx] are fx, fy , cx , cy. fx and fy are the focal lengths of the camera expressed in pixel units.
# These values describe how much the camera magnifies the scene in terms of pixels per unit distance in the x and y directions, respectively.
# cx​ and cy​: These are the coordinates of the optical center (also known as the principal point) of the camera, expressed in pixel coordinates. 
# For your matrix, cx=572.83382086 and cy​=324.73669189.
# This point is ideally the center of the image sensor but can vary based on manufacturing variances or sensor alignme