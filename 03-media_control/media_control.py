import cv2
import sys
from classifier_gesture import GestureClassifier
from pynput.keyboard import Key, Controller
from config import CONDITIONS, WINDOW_HEIGHT, WINDOW_NAME, WINDOW_WIDTH

video_id = 0
keyboard = Controller()

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# get most common element
# https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
def most_frequent(list):
    counter = 0
    num = list[0]

    for i in list:
        curr_frequency = list.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num

class MediaController:

    def __init__(self):
        self.cap = cv2.VideoCapture(video_id)
        self.classifier = GestureClassifier()
        self.predictions = []
        cv2.namedWindow(WINDOW_NAME)

    def control_media(self):
        while True:
            success, frame = self.cap.read()
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)

            if success:
                prediction = self.classifier.predict(frame)
                self.predictions.append(prediction)

                #of most common of previous 10 elements is a viable class input the responding keyevent
                if len(self.predictions) > 10:
                    mostCommonPrediction = most_frequent(self.predictions)

                    if mostCommonPrediction == CONDITIONS[0]:
                        keyboard.press(Key.media_play_pause)
                    elif mostCommonPrediction == CONDITIONS[2]:
                        keyboard.press(Key.media_volume_up)
                    elif mostCommonPrediction == CONDITIONS[3]:
                        keyboard.press(Key.media_volume_down)

                    self.predictions = []

                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()


mediacontroller = MediaController()
mediacontroller.control_media()
