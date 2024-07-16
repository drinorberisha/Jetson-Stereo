import cv2
import numpy as np
import threading
# commenting
class Start_Cameras:

    def __init__(self, sensor_id):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        self.sensor_id = sensor_id

        gstreamer_pipeline_string = self.gstreamer_pipeline()
        self.open(gstreamer_pipeline_string)

    def open(self, gstreamer_pipeline_string):
        gstreamer_pipeline_string = self.gstreamer_pipeline()
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            grabbed, frame = self.video_capture.read()
            print("Cameras are opened")

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)
            return

        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None

        if self.video_capture is not None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()

    def updateCamera(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.read_thread is not None:
            self.read_thread.join()

    def gstreamer_pipeline(self,
            sensor_mode=3,
            capture_width=1280,
            capture_height=720,
            display_width=640,
            display_height=360,
            framerate=30,
            flip_method=0,
    ):
        return (
            "nvarguscamerasrc sensor-id={} sensor-mode={} ! "
            "video/x-raw(memory:NVMM), "
            "width=(int){}, height=(int){}, "
            "format=(string)NV12, framerate=(fraction){}/1 ! "
            "nvvidconv flip-method={} ! "
            "video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink".format(
                self.sensor_id,
                sensor_mode,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height
            )
        )

def correct_color_balance(image):
    b, g, r = cv2.split(image)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)
    return cv2.merge((b, g, r))

if __name__ == "__main__":
    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()

    while True:
        left_grabbed, left_frame = left_camera.read()
        right_grabbed, right_frame = right_camera.read()
    
        if left_grabbed and right_grabbed:
            left_frame = correct_color_balance(left_frame)
            right_frame = correct_color_balance(right_frame)

            images = np.hstack((left_frame, right_frame))
            cv2.imshow("Camera Images", images)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break
        else:
            break

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()
