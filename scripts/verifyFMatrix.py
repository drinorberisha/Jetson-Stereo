import cv2
import numpy as np
import glob
import os

# Load calibration data
calibration_data = np.load('../data/stereo_calibration_data.npz')
F = calibration_data['F']

# Directory containing valid image pairs and corner points
valid_image_dir = '../images/valid_pairs'

def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    ''' Draw epipolar lines on the images'''
    r, c = img1.shape[:2]
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(map(int, pt1.ravel())), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(map(int, pt2.ravel())), 5, color, -1)
    return img1_color, img2_color


def verify_fundamental_matrix(F, pts1, pts2):
    ''' Verify the Fundamental Matrix by checking the epipolar constraint '''
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    # Epipolar constraint: pts2_h.T * F * pts1_h should be close to 0
    errors = []
    for i in range(pts1.shape[0]):
        error = np.abs(np.dot(pts2_h[i], np.dot(F, pts1_h[i].T)))
        errors.append(error)
    mean_error = np.mean(errors)
    return mean_error

# Process images from 1 to 20
for i in range(1, 21):
    left_image_path = os.path.join(valid_image_dir, f'Im_L_{i}.png')
    right_image_path = os.path.join(valid_image_dir, f'Im_R_{i}.png')
    corners_file = os.path.join(valid_image_dir, f'corners_Im_L_{i}.npz')

    if not (os.path.exists(left_image_path) and os.path.exists(right_image_path) and os.path.exists(corners_file)):
        print(f"Skipping image pair {i} as files are missing.")
        continue

    img1 = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    corner_points = np.load(corners_file)
    points1 = corner_points['corners_left'].reshape(-1, 2)
    points2 = corner_points['corners_right'].reshape(-1, 2)

    # Verify Fundamental Matrix
    mean_error = verify_fundamental_matrix(F, points1, points2)
    print(f"Image pair {i}: Mean Epipolar Constraint Error: {mean_error}")

    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_lines, img2_points = draw_epipolar_lines(img1, img2, lines1, points1, points2)

    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_lines, img1_points = draw_epipolar_lines(img2, img1, lines2, points2, points1)

    # Save the images with epipolar lines
    cv2.imwrite(os.path.join(valid_image_dir, f'epipolar_left_{i}.png'), img1_lines)
    cv2.imwrite(os.path.join(valid_image_dir, f'epipolar_right_{i}.png'), img2_lines)

print("Processing complete. Epipolar line images have been saved.")