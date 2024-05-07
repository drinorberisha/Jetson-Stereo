import cv2
import time

cap = cv2.VideoCapture(0)  # Adjust the device number if needed

print("Starting live capture. Press 's' to save the frame, 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                filename = f"../images/saved_frames/chessboard_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('q'):
                print("Quitting live capture.")
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
