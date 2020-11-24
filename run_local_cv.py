# Run this file on CV2 in local machine to construct a Concentration Index (CI).
# Video image will show emotion on first line, and engagement on second. Engagement/concentration classification displays either 'Pay attention', 'You are engaged' and 'you are highly engaged' based on CI. Webcam is required.
# Analysis is in 'Util' folder.
import cv2
import numpy as np
from util.analysis_realtime import analysis

# Initializing
ana = analysis()

########################
# Realtime

cap = cv2.VideoCapture(0)
# Capture every frame and send to detector
while True:
    _, frame = cap.read()
    bm = ana.detect_face(frame)

    cv2.imshow("Frame", bm)

    key = cv2.waitKey(1)
# Exit if 'q' is pressed
    if key == ord('q'):
        break
# Release the memory
cap.release()
cv2.destroyAllWindows()

#########################
def get_output_file(path, fps=30):
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
        frameSize=(display_width, display_height),
        isColor=True,
    )

source = "/Users/alexsun/Desktop/Research_CV/distractionModel/zoom_class_2.mp4"
output_path = "/Users/alexsun/Desktop/Research_CV/distractionModel/zoom_class_2_out.mp4"
cap = cv2.VideoCapture(source)
display_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
display_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fps = cap.get(cv2.CAP_PROP_FPS)
print(output_fps)
output_file = get_output_file(output_path, 5)
# output_file = get_output_file(output_path, 10)
cnt = 0

while cap.isOpened():
    ret, frame = cap.read()
    print(ret)
    if ret:
        bm = ana.detect_face(frame)
        output_file.write(frame)
        cnt += 10
        cap.set(1,cnt)
    else:
        cap.release()
        break
output_file.release()