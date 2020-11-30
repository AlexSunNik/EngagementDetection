# Module of CELA
# Author: Xinglong Sun
# Based on https://github.com/CaedenZ/distractionModel
from util.analysis_realtime import analysis
import cv2
import numpy as np

class EngagementModel:
    # The top level model for engagement detection
    # mode = 0 for realtime
    # mode = 1 for processing/analysis
    def __init__(self, mode, input_file="", output_file=""):
        self.ana = analysis()
        self.mode = mode
        if mode == 0:
            self.cap = cv2.VideoCapture(0)
        if mode == 1:
            self.source = input_file
            self.output_path = output_file
            self.cap = cv2.VideoCapture(self.source)
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 5 is the output fps
            self.output_file = self.get_output_file(self.output_path, 5)

            self.output_fps = self.cap.get(cv2.CAP_PROP_FPS)
    def predict(self):
        # Real-time
        if self.mode == 0:
            print("Starting real-time detection.")
            # Capture every frame and send to detector
            while True:
                _, frame = self.cap.read()
                bm = self.ana.detect_face(frame)
                cv2.imshow("Frame", bm)
                key = cv2.waitKey(1)
                # Exit if 'q' is pressed
                if key == ord('q'):
                    break
            # Clear
            self.cap.release()
            cv2.destroyAllWindows()

        elif self.mode == 1:
            print("Starting processing.")
            cnt = 0
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    bm = self.predict_image(frame)
                    self.output_file.write(bm)
                    cnt += 10
                    self.cap.set(1,cnt)
                else:
                    self.cap.release()
                    break
            self.output_file.release()
    
    def predict_image(self, image):
        return self.ana.detect_face(image)
    
    #########################
    def get_output_file(self, path, fps=30):
        """
        Return a video writer object.
        Args:
            path (str): path to the output video file.
            fps (int or float): frames per second.
        """
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(fps),
            frameSize=(self.display_width, self.display_height),
            isColor=True,
        )

# Testing
# Realtime
# model = EngagementModel(0)
# model.predict()

# Video
# model = EngagementModel(1, "zoom_class_1.mp4", "zoom_class_1_out.mp4")
# model.predict()

# Image
# model = EngagementModel(1, "", "")
# img = cv2.imread("face1.jpg")
# out = model.predict_image(img)
# cv2.imshow('output', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()