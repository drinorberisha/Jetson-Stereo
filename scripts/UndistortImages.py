import cv2
import numpy as np

# Load the previously saved calibration data
print("Loading calibration data...")
data = np.load('../data/calibration_data_right.npz')
mtx = data['mtx']
dist = data['dist']

# Load an image
img_path = '../images/data/imgs/rightcamera/Im_R_5.png'
print(f"Loading image {img_path} for undistortion...")
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Obtain the new camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Save or display the undistorted image
output_path = '../results/undistorted_images/undistorted_image.jpg'
cv2.imwrite(output_path, dst)
print(f"Undistorted image saved as {output_path}")

cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
