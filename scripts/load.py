import numpy as np

def load_calibration_data(calibration_file):
    calibration_data = np.load(calibration_file)
    mtx_left = calibration_data['mtx_left']
    dist_left = calibration_data['dist_left']
    mtx_right = calibration_data['mtx_right']
    dist_right = calibration_data['dist_right']
    R = calibration_data['R']
    T = calibration_data['T']
    R1 = calibration_data['R1']
    R2 = calibration_data['R2']
    P1 = calibration_data['P1']
    P2 = calibration_data['P2']
    Q = calibration_data['Q']
    
    print("Left Camera Matrix:\n", mtx_left)
    print("Left Distortion Coefficients:\n", dist_left)
    print("Right Camera Matrix:\n", mtx_right)
    print("Right Distortion Coefficients:\n", dist_right)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
    print("Left Rectification Matrix:\n", R1)
    print("Right Rectification Matrix:\n", R2)
    print("Left Projection Matrix:\n", P1)
    print("Right Projection Matrix:\n", P2)
    print("Disparity-to-depth Mapping Matrix:\n", Q)

load_calibration_data('../data/stereo_calibration_data.npz')
