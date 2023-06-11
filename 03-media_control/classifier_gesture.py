from keras.models import load_model
import numpy as np
import cv2
from config import CONDITIONS

# number of color channels we want to use
# set to 1 to convert to grayscale
# set to 3 to use color images
COLOR_CHANNELS = 3


class GestureClassifier:

    def __init__(self):
        self.model = load_model('pretrained_model')

    def predict(self, img):
        # Preprocess image
        img = cv2.resize(img, (64, 64))
        img = np.array(img).astype('float32')
        img = img / 255.
        img = img.reshape(-1, 64, 64, COLOR_CHANNELS)

        # predict class
        pred = self.model.predict(img, verbose=0)
        pred = np.argmax(pred, axis=1)
        return CONDITIONS[pred[0]]
