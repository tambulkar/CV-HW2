import cv2
import numpy as np
from skimage.feature import match_template

def get_frames(path, grayscale=True):
    vid = cv2.VideoCapture(path)
    frames = []
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)#cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))

    return np.array(frames)
