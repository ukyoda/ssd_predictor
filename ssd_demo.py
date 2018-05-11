#-*- coding: utf-8 -*-

# SSDのデモプログラム

import os
import cv2
import numpy as np
import argparse

from predictor.ssd.SSDPredictor import SSDPrecitor

ROOT_DIR = os.path.dirname(__file__)

def main(args=None):
    # ビデオロード

    if args.camera:
        video_resource = args.camera_select
    else:
        video_resource = args.file

    video = cv2.VideoCapture(video_resource)
    if not video.isOpened:
        raise IOError('読み込みエラー')

    # 識別器作成
    modelfile = os.path.join(ROOT_DIR, 'models/weights_SSD300.hdf5')
    predictor = SSDPrecitor(modelfile)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='動画ファイルを指定(カメラモードの時は見指定でOK)')
    parser.add_argument('--camera', action='store_true', help='指定するとカメラを起動する')
    parser.add_argument('--camera_select', type=int, default=0, help='カメラを指定した時、カメラデバイスのIDを指定')
    args = parser.parse_args()
    if not args.file and not args.camera:
        parser.print_help()
    else:
        main(args=args)
