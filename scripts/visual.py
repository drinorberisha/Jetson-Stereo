import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the calibration data
data = np.load('../data/stereo_calibration_data.npz')

# Extract relevant data
mtx_left = data['mtx_left']
dist_left = data['dist_left']
mtx_right = data['mtx_right']
dist_right = data['dist_right']
R = data['R']
T = data['T']

# Print camera matrices and distortion coefficients
print("Left Camera Matrix:\n", mtx_left)
print("Left Distortion Coefficients:\n", dist_left)
print("Right Camera Matrix:\n", mtx_right)
print("Right Distortion Coefficients:\n", dist_right)

# Setup a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Origin
ax.scatter(0, 0, 0, color='r', label='Left Camera (Origin)')

# Translation vector from left to right camera
ax.quiver(0, 0, 0, T[0], T[1], T[2], color='b', label='Translation Vector (Right Camera)')

# Set limits and labels
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Rotation matrix as a directional plot
x, y, z = np.eye(3)
u, v, w = R @ [x, y, z]
ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], u, v, w, color=['r', 'g', 'b'])

# Legend and plot
ax.legend()
plt.title('Camera Translation and Rotation Visualization')
plt.show()
