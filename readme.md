
### README.md

# Stereo Calibration and Rectification on Jetson Nano

This project is designed to run stereo calibration and rectification processes on a Jetson Nano embedded system using Python 2.7. It involves capturing stereo images, calibrating the cameras, and rectifying the images.

## Prerequisites

Ensure the following software is installed on your Jetson Nano:
- Python 2.7
- OpenCV (compatible with Python 2.7)
- NumPy

## Setup

### Step 1: Create a Virtual Environment

First, create a virtual environment using `venv` and activate it.

```bash
# Create a virtual environment
python2.7 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate
```

### Step 2: Install Required Packages

With the virtual environment activated, install the required Python packages.

```bash
pip install numpy opencv-python==3.4.11.45
```

### Step 3: Run the Master Script

Now you can run the master script which orchestrates the process of capturing images and performing calibration.

```bash
python master_script.py
```

## Script Documentation

### master_script.py

This script orchestrates the entire process by calling other scripts sequentially.

```python
import os

def run_script(script_name):
    os.system(f"python2.7 {script_name}")

def main():
    print("Starting process...")
    run_script('scripts/LiveCapture.py')
    run_script('scripts/JulyCal.py')
    # run_script('scripts/AdaptiveHistogramEqualization.py')
    # run_script('scripts/UndistortImages.py')
    # run_script('scripts/Rectification.py')
    # run_script('scripts/sgmb3Cuda.py')
    print("Process completed!")

if __name__ == "__main__":
    main()
```

#### Functions

- `run_script(script_name)`: Executes the given script using `python2.7`.
- `main()`: Calls the `LiveCapture.py` script to capture images and then `JulyCal.py` to perform stereo calibration.

### scripts/LiveCapture.py

This script captures stereo images and saves them in the specified directory.

```python
import cv2

def capture_images():
    cap_left = cv2.VideoCapture(0)  # Adjust the index if needed
    cap_right = cv2.VideoCapture(1)  # Adjust the index if needed

    if not (cap_left.isOpened() and cap_right.isOpened()):
        print("Error: Could not open video streams.")
        return

    for i in range(20):  # Capture 20 pairs of images
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not (ret_left and ret_right):
            print("Error: Could not read frames.")
            continue
        
        combined_image = cv2.hconcat([frame_left, frame_right])
        cv2.imwrite(f'../images/saved_frames/stereo_image_{i}.jpg', combined_image)
        print(f"Captured stereo_image_{i}.jpg")
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
```

#### Functions

- `capture_images()`: Captures 20 pairs of stereo images from two cameras and saves them as combined images.


#### Functions

- `split_stereo_image(stereo_image)`: Splits a stereo image into left and right images.
- `find_chessboard_corners(image, board_size)`: Finds and refines chessboard corners in the image.
- `stereo_calibrate(stereo_images, board_size, square_size)`: Performs stereo calibration using the captured images and returns the calibration parameters.

## Notes

- Ensure that the cameras are correctly connected and configured before running the scripts.
- The calibration process requires at least 10 good stereo image pairs with a visible chessboard pattern for accurate results.

This should cover the setup and usage of the scripts for stereo calibration on a Jetson Nano running Python 2.7.