from cv2 import cv2
import YOLO
import Detect
capture = cv2.VideoCapture(0)

if capture.isOpened() is False:
    print("Error opening camera")
ret, frame = capture.read()
if ret is True:
    detected = YOLO.humanDetect(frame)

    if detected is not None:

        # cv2.imshow("human detected", detected)
        # cv2.waitKey()
        cv2.imshow("Person Detected", detected)
        cv2.waitKey()
        print("Human Detected through YOLO")
        face = Detect.detect_face(detected)
        mask = Detect.detect_mask(face)
        print(mask)

    else:
        print('No human in the frame')
capture.release()
cv2.destroyAllWindows()
