import cv2
import numpy as np
import glob
import os

def split_stereo_image(stereo_image):
    if stereo_image is None or len(stereo_image) == 0:
        print("Error: Stereo image is empty or None")
        return None, None

    height, width = stereo_image.shape[:2]
    midpoint = width // 2
    left_img = stereo_image[:, :midpoint]
    right_img = stereo_image[:, midpoint:]
    return left_img, right_img

def compute_matching_homographies(e2, F, im2, points1, points2):
    '''
    Compute the matching homography matrices
    '''
    h, w = im2.shape[:2]
    # create the homography matrix H2 that moves the epipole to infinity

    # create the translation matrix to shift to the image center
    T = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])
    e2_p = T @ e2
    e2_p = e2_p / e2_p[2]
    e2x = e2_p[0]
    e2y = e2_p[1]
    # create the rotation matrix to rotate the epipole back to X axis
    if e2x >= 0:
        a = 1
    else:
        a = -1
    R1 = a * e2x / np.sqrt(e2x ** 2 + e2y ** 2)
    R2 = a * e2y / np.sqrt(e2x ** 2 + e2y ** 2)
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
    e2_p = R @ e2_p
    x = e2_p[0]
    # create matrix to move the epipole to infinity
    G = np.array([[1, 0, 0], [0, 1, 0], [-1/x, 0, 1]])
    # create the overall transformation matrix
    H2 = np.linalg.inv(T) @ G @ R @ T

    # create the corresponding homography matrix for the other image
    e_x = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    M = e_x @ F + e2.reshape(3, 1) @ np.array([[1, 1, 1]])
    
    # Convert points to homogeneous coordinates
    points1_h = np.hstack([points1, np.ones((points1.shape[0], 1))])
    points2_h = np.hstack([points2, np.ones((points2.shape[0], 1))])

    points1_t = H2 @ M @ points1_h.T
    points2_t = H2 @ points2_h.T
    
    points1_t /= points1_t[2, :]
    points2_t /= points2_t[2, :]
    
    b = points2_t[0, :]
    a = np.linalg.lstsq(points1_t.T, b, rcond=None)[0]
    H_A = np.array([a, [0, 1, 0], [0, 0, 1]])
    H1 = H_A @ H2 @ M
    return H1, H2

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
F = calibration_data['F']
valid_roi1 = calibration_data['valid_roi1']
valid_roi2 = calibration_data['valid_roi2']
img_size = tuple(calibration_data['img_size'])

print(f"Calibration data loaded. \nmtx_left: {mtx_left}\nmtx_right: {mtx_right}\nF: {F}\n")

# Directory containing the saved images
images_dir = '../images/valid_pairs'
rectified_dir = '../images/rectified_frames'
if not os.path.exists(rectified_dir):
    os.makedirs(rectified_dir)

# Load images
image_files = sorted(glob.glob(os.path.join(images_dir, 'valid_*.jpg')))
if not image_files:
    print("No images found in the directory.")
    exit()

for image_file in image_files:
    image = cv2.imread(image_file)
    if image is None:
        print(f"Failed to load image {image_file}")
        continue

    left_frame, right_frame = split_stereo_image(image)
    if left_frame is None or right_frame is None:
        print(f"Failed to split image {image_file}")
        continue

    # Detect keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left_frame, None)
    kp2, des2 = sift.detectAndCompute(right_frame, None)

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 8:  # Minimum number of matches required for findFundamentalMat
            points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            e2 = np.linalg.svd(F)[-1][-1]
            e2 /= e2[2]
            H1, H2 = compute_matching_homographies(e2, F, right_frame, points1, points2)

            rect_left = cv2.warpPerspective(left_frame, H1, img_size)
            rect_right = cv2.warpPerspective(right_frame, H2, img_size)

            # Debugging: Check if rectification maps are applied correctly
            if rect_left is None or rect_right is None:
                print(f"Rectification failed for image {image_file}")
                continue

            # Debugging: Print valid ROI values
            print(f"Valid ROI1: {valid_roi1}")
            print(f"Valid ROI2: {valid_roi2}")

            # Crop the rectified images
            x1, y1, w1, h1 = valid_roi1
            x2, y2, w2, h2 = valid_roi2

            # Check if cropping dimensions are valid
            if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
                print(f"Invalid cropping dimensions for image {image_file}")
                continue

            rect_left = rect_left[y1:y1+h1, x1:x1+w1]
            rect_right = rect_right[y2:y2+h2, x2:x2+w2]

            # Check if the images are empty after cropping
            if rect_left.size == 0:
                print(f"Left rectified image is empty after cropping for image {image_file}")
                continue
            if rect_right.size == 0:
                print(f"Right rectified image is empty after cropping for image {image_file}")
                continue

            # Save rectified images
            basename = os.path.basename(image_file)
            left_rectified_path = os.path.join(rectified_dir, "rectified_left_{}".format(basename))
            right_rectified_path = os.path.join(rectified_dir, "rectified_right_{}".format(basename))

            cv2.imwrite(left_rectified_path, rect_left)
            cv2.imwrite(right_rectified_path, rect_right)

            # Display the rectified images
            combined_rectified = np.hstack((rect_left, rect_right))
            cv2.imshow("Rectified Images", combined_rectified)

            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                break
        else:
            print(f"Not enough good matches found for image {image_file}")
    else:
        print(f"Failed to detect descriptors in one of the frames for image {image_file}")

cv2.destroyAllWindows()
