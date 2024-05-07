import cv2
import numpy as np
import sys

print("Initializing CUDA streams for asynchronous operations...")
stream = cv2.cuda_Stream()

if len(sys.argv) > 2:
    left_img_path = sys.argv[1]
    right_img_path = sys.argv[2]
else:
    print("Usage: python script.py left_image.png right_image.png")
    sys.exit(1)

print("Loading stereo images into GPU memory...")
left_img_gpu = cv2.cuda.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
right_img_gpu = cv2.cuda.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

print("Applying bilateral filter to reduce noise and preserve edges...")
left_img_filtered = cv2.cuda.bilateralFilter(left_img_gpu, 9, 75, 75, stream=stream)
right_img_filtered = cv2.cuda.bilateralFilter(right_img_gpu, 9, 75, 75, stream=stream)

# If Gaussian blur is needed, uncomment the following lines:
# print("Applying Gaussian blur...")
# left_img_blurred = cv2.cuda.GaussianBlur(left_img_filtered, (5, 5), 0, stream=stream)
# right_img_blurred = cv2.cuda.GaussianBlur(right_img_filtered, (5, 5), 0, stream=stream)

print("Setting up CUDA StereoSGBM with optimized parameters...")
stereo = cv2.cuda.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Adjust accordingly
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=50,
    speckleRange=16,  # Adjusted for underwater conditions
    preFilterCap=63,
    mode=cv2.cuda.STEREO_SGBM_MODE_SGBM_3WAY
)

print("Computing disparity map using GPU-accelerated StereoSGBM...")
disparity_gpu = stereo.compute(left_img_filtered, right_img_filtered, stream=stream)
disparity = disparity_gpu.download()  # Download disparity map from GPU to CPU memory for further processing

print("Refining disparity map using Weighted Least Squares Filter...")
lmbda = 80000
sigma = 1.2
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
disparity_filtered = wls_filter.filter(disparity, left_img_gpu.download(), None, right_img_gpu.download())

print("Normalizing and preparing disparity map for visualization...")
disparity_filtered_safe = np.nan_to_num(disparity_filtered, nan=0.0, posinf=255, neginf=0)
disparity_filtered_clipped = np.clip(disparity_filtered_safe, 0, 255).astype(np.uint8)
disparity_filtered_normalized = cv2.normalize(disparity_filtered_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

output_path = 'results/disparity_maps/refined_disparity.png'
cv2.imwrite(output_path, disparity_filtered_normalized)
print(f"Refined disparity map saved as {output_path}")

print("Displaying refined disparity map...")
cv2.imshow('Refined Disparity', disparity_filtered_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Processing complete. Closing application.")
