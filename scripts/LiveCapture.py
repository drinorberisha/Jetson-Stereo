import cv2
import numpy as np
from start_cameras import Start_Cameras  # Ensure this module is correctly implemented
from datetime import datetime
import time
import os
from os import path

# Photo Taking presets
total_photos = 20  # Number of images to take
countdown = 2 # Interval for count-down timer, seconds
font = cv2.FONT_HERSHEY_SIMPLEX  # Countdown timer font

def TakePictures():
    val = input("Would you like to start the image capturing? (Y/N) ")

    if val.lower() == "y":
        left_camera = Start_Cameras(0).start()
        right_camera = Start_Cameras(1).start()
        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)

        counter = 0
        t2 = datetime.now()
        images_dir = '../images/saved_frames'  # Updated directory to match your existing setup
        if not path.isdir(images_dir):
            os.makedirs(images_dir)
            print(f"Directory created: {images_dir}")

        while counter < total_photos:  # Changed <= to < for correct counting
            t1 = datetime.now()
            countdown_timer = countdown - int((t1 - t2).total_seconds())

            left_grabbed, left_frame = left_camera.read()
            right_grabbed, right_frame = right_camera.read()

            if left_grabbed and right_grabbed:
                images = np.hstack((left_frame, right_frame))
                if countdown_timer == -1:
                    counter += 1
                    filename = f"{images_dir}/jetsonchessboard_{counter:02d}.jpg"  # Formatted filename
                    cv2.imwrite(filename, images)
                    print(f"Image: {filename} is saved!")
                    t2 = datetime.now()
                    time.sleep(1)
                    countdown_timer = 0

                cv2.putText(images, str(countdown_timer), (50, 50), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow("Images", images)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to capture images from both cameras.")
                break

    elif val.lower() == "n":
        print("Quitting!")
        exit()
    else:
        print("Please try again!")

    left_camera.release()
    right_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TakePictures()
