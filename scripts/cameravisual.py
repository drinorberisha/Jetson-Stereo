import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_intrinsic_parameters(mtx_left, mtx_right, output_folder):
    # Extract focal lengths and principal points
    fx_left, fy_left = mtx_left[0, 0], mtx_left[1, 1]
    cx_left, cy_left = mtx_left[0, 2], mtx_left[1, 2]
    fx_right, fy_right = mtx_right[0, 0], mtx_right[1, 1]
    cx_right, cy_right = mtx_right[0, 2], mtx_right[1, 2]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Left camera
    ax1.arrow(cx_left, cy_left, fx_left/100, 0, head_width=20, head_length=20, fc='r', ec='r')
    ax1.arrow(cx_left, cy_left, 0, fy_left/100, head_width=20, head_length=20, fc='g', ec='g')
    ax1.plot(cx_left, cy_left, 'bo', markersize=10)
    ax1.set_xlim(0, mtx_left[0, 2]*2)
    ax1.set_ylim(mtx_left[1, 2]*2, 0)  # Invert y-axis to match image coordinates
    ax1.set_title('Left Camera Intrinsic Parameters')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.legend(['Principal Point', 'fx', 'fy'])

    # Right camera
    ax2.arrow(cx_right, cy_right, fx_right/100, 0, head_width=20, head_length=20, fc='r', ec='r')
    ax2.arrow(cx_right, cy_right, 0, fy_right/100, head_width=20, head_length=20, fc='g', ec='g')
    ax2.plot(cx_right, cy_right, 'bo', markersize=10)
    ax2.set_xlim(0, mtx_right[0, 2]*2)
    ax2.set_ylim(mtx_right[1, 2]*2, 0)  # Invert y-axis to match image coordinates
    ax2.set_title('Right Camera Intrinsic Parameters')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.legend(['Principal Point', 'fx', 'fy'])

    plt.tight_layout()
    plt.savefig(f'{output_folder}/intrinsic_parameters.png')
    plt.close()

def visualize_extrinsic_parameters(R, T, output_folder):
    print("Debug: R shape:", R.shape)
    print("Debug: R type:", type(R))
    print("Debug: R content:", R)
    print("Debug: T shape:", T.shape if isinstance(T, np.ndarray) else "T is not a numpy array")
    print("Debug: T type:", type(T))
    print("Debug: T content:", T)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the left camera (origin)
    ax.scatter(0, 0, 0, c='r', s=100)
    ax.quiver(0, 0, 0, 1, 0, 0, length=0.5, normalize=True, color='r')
    ax.quiver(0, 0, 0, 0, 1, 0, length=0.5, normalize=True, color='g')
    ax.quiver(0, 0, 0, 0, 0, 1, length=0.5, normalize=True, color='b')

    # Plot the right camera
    if isinstance(T, np.ndarray) and T.shape == (3,1):
        T = T.flatten()
    elif isinstance(T, np.ndarray) and T.shape == (1,3):
        T = T.flatten()
    elif not isinstance(T, np.ndarray):
        T = np.array(T)

    if T.shape != (3,):
        print(f"Error: Unexpected shape for T: {T.shape}")
        return

    ax.scatter(T[0], T[1], T[2], c='b', s=100)
    ax.quiver(T[0], T[1], T[2], R[0, 0], R[1, 0], R[2, 0], length=0.5, normalize=True, color='r')
    ax.quiver(T[0], T[1], T[2], R[0, 1], R[1, 1], R[2, 1], length=0.5, normalize=True, color='g')
    ax.quiver(T[0], T[1], T[2], R[0, 2], R[1, 2], R[2, 2], length=0.5, normalize=True, color='b')

    # Connect cameras with a line
    ax.plot([0, T[0]], [0, T[1]], [0, T[2]], 'k--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Extrinsic Parameters: Camera Positions and Orientations')
    
    # Set equal aspect ratio
    ax.set_box_aspect((np.ptp(T), np.ptp(T), np.ptp(T)))

    plt.savefig(f'{output_folder}/extrinsic_parameters.png')
    plt.close()

def visualize_camera_parameters(calibration_data, output_folder):
    # Extract relevant data
    mtx_left = calibration_data['mtx_left']
    mtx_right = calibration_data['mtx_right']
    R = calibration_data['R']
    T = calibration_data['T']

    print("Debug: mtx_left shape:", mtx_left.shape)
    print("Debug: mtx_right shape:", mtx_right.shape)
    print("Debug: R shape:", R.shape)
    print("Debug: T shape:", T.shape if isinstance(T, np.ndarray) else "T is not a numpy array")

    # Visualize intrinsic parameters
    visualize_intrinsic_parameters(mtx_left, mtx_right, output_folder)

    # Visualize extrinsic parameters
    visualize_extrinsic_parameters(R, T, output_folder)
def main():
    # Input and output folders
    calibration_file = '../data/stereo_calibration_data.npz'
    visualization_output_folder = '../visualizations'

    # Load calibration data
    calibration_data = np.load(calibration_file)
    print(f"Calibration data loaded from {calibration_file}")
    
    print("Debug: Keys in calibration_data:", calibration_data.keys())

    # Visualize camera parameters
    visualize_camera_parameters(calibration_data, visualization_output_folder)

    print("Camera parameter visualization complete.")

if __name__ == "__main__":
    main()