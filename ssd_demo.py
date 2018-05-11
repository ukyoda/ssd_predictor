#-*- coding: utf-8 -*-

# SSDのデモプログラム

import os
import cv2
import numpy as np

from predictor.ssd.SSDPredictor import SSDPrecitor

ROOT_DIR = os.path.dirname(__file__)

def main(args=None):
    modelfile = os.path.join(ROOT_DIR, 'models/weights_SSD300.hdf5')
    predictor = SSDPrecitor(modelfile)
    video = cv2.VideoCapture(0)
    if not video.isOpened:
        raise IOError('エラー')

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = predictor.predict(frame)
        for label_id, score, box in results:
            left = (int(box[0]), int(box[1]))
            right = (int(box[2]), int(box[3]))
            print(label_id, score)
            cv2.rectangle(frame, left, right, (0, 0, 255), 2)
        cv2.imshow('result', frame)
        key = cv2.waitKey(10) & 0xff
        if key == ord('q') or key == 27:
            break

if __name__ == '__main__':
    main()
