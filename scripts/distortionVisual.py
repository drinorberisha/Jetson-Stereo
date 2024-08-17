import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import cv2

def generate_distortion_heatmap(camera_matrix, dist_coeffs, img_size):
    # Create a grid of points
    x, y = np.meshgrid(np.linspace(0, img_size[0], 20),
                       np.linspace(0, img_size[1], 20))
    
    # Combine x and y into a single array of points
    points = np.column_stack((x.ravel(), y.ravel()))
    
    # Undistort the points
    undistorted_points = cv2.undistortPoints(points.reshape(-1, 1, 2), 
                                             camera_matrix, 
                                             dist_coeffs, 
                                             P=camera_matrix)
    
    # Calculate the distortion magnitude
    distortion = np.linalg.norm(points - undistorted_points.reshape(-1, 2), axis=1)
    
    # Reshape the distortion array to match the original grid shape
    distortion_map = distortion.reshape(x.shape)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(distortion_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Distortion Magnitude (pixels)')
    plt.title('Camera Distortion Heatmap')
    plt.xlabel('Image Width')
    plt.ylabel('Image Height')
    
    return plt

def visualize_distortion(calibration_data, output_folder):
    # Load calibration data
    mtx_left = calibration_data['mtx_left']
    dist_left = calibration_data['dist_left']
    mtx_right = calibration_data['mtx_right']
    dist_right = calibration_data['dist_right']
    img_size = calibration_data['img_size']

    # Generate and save left camera distortion heatmap
    plt_left = generate_distortion_heatmap(mtx_left, dist_left, img_size)
    plt_left.savefig(os.path.join(output_folder, 'left_camera_distortion_heatmap.png'))
    plt_left.close()

    # Generate and save right camera distortion heatmap
    plt_right = generate_distortion_heatmap(mtx_right, dist_right, img_size)
    plt_right.savefig(os.path.join(output_folder, 'right_camera_distortion_heatmap.png'))
    plt_right.close()

    print("Distortion heatmaps generated and saved.")

def analyze_calibration_results(calibration_data, output_folder):
    # Extract relevant data
    mtx_left = calibration_data['mtx_left']
    mtx_right = calibration_data['mtx_right']
    dist_left = calibration_data['dist_left']
    dist_right = calibration_data['dist_right']
    R = calibration_data['R']
    T = calibration_data['T']

    # Analyze and visualize focal lengths
    focal_lengths = {
        'Left X': mtx_left[0, 0],
        'Left Y': mtx_left[1, 1],
        'Right X': mtx_right[0, 0],
        'Right Y': mtx_right[1, 1]
    }
    plt.figure(figsize=(10, 6))
    plt.bar(focal_lengths.keys(), focal_lengths.values())
    plt.title('Comparison of Focal Lengths')
    plt.ylabel('Focal Length (pixels)')
    plt.savefig(os.path.join(output_folder, 'focal_length_comparison.png'))
    plt.close()

    # Analyze and visualize distortion coefficients
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.bar(range(len(dist_left[0])), dist_left[0])
    plt.title('Left Camera Distortion Coefficients')
    plt.subplot(122)
    plt.bar(range(len(dist_right[0])), dist_right[0])
    plt.title('Right Camera Distortion Coefficients')
    plt.savefig(os.path.join(output_folder, 'distortion_coefficients.png'))
    plt.close()

    # Visualize camera position
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, R[0, 0], R[1, 0], R[2, 0], length=0.1, normalize=True, color='r')
    ax.quiver(0, 0, 0, R[0, 1], R[1, 1], R[2, 1], length=0.1, normalize=True, color='g')
    ax.quiver(0, 0, 0, R[0, 2], R[1, 2], R[2, 2], length=0.1, normalize=True, color='b')
    ax.quiver(0, 0, 0, T[0], T[1], T[2], length=0.1, normalize=True, color='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Relative Camera Positions')
    plt.savefig(os.path.join(output_folder, 'camera_positions.png'))
    plt.close()

    print("Additional analysis and visualizations completed.")

def main():
    # Input and output folders
    calibration_file = '../data/stereo_calibration_data.npz'
    visualization_output_folder = '../visualizations'

    # Ensure output folder exists
    os.makedirs(visualization_output_folder, exist_ok=True)

    # Load calibration data
    calibration_data = np.load(calibration_file)
    print(f"Calibration data loaded from {calibration_file}")

    # Generate and save distortion heatmaps
    visualize_distortion(calibration_data, visualization_output_folder)

    # Perform additional analysis and visualizations
    analyze_calibration_results(calibration_data, visualization_output_folder)

    print("Analysis and visualization complete.")

if __name__ == "__main__":
    main()