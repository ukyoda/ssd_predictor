#-*- coding: utf-8 -*-

# SSDのデモプログラム

import os
import cv2
import numpy as np
import argparse

from predictor.ssd.SSDPredictor import SSDPrecitor
from predictor.utility import create_class_colors, create_classnames

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
    predictor = SSDPrecitor(modelfile, num_classes=args.num_classes, conf_thresh=args.thresh)
    # 色
    colors = create_class_colors(args.num_classes)
    # ラベル
    if args.labelfile:
        labels = create_classnames(filepath=args.labelfile)
    else:
        labels = create_classnames(num_classes=args.num_classes)


    while True:
        ret, frame = video.read()
        if not ret:
            break

        # オブジェクト検出を行う
        results = predictor.predict(frame)

        # 検出結果を描画
        for label_id, score, box in results:
            label = labels[label_id]
            color = colors[label_id]
            # 領域を描画
            left = (int(box[0]), int(box[1]))
            right = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, left, right, color, 2)

            # ラベル描画
            text_pos = (left[0], left[1]-10)
            cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        cv2.imshow('result', frame)
        key = cv2.waitKey(10) & 0xff
        if key == ord('q') or key == 27:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='動画ファイルを指定(カメラモードの時は未指定でOK)')
    parser.add_argument('--camera', action='store_true', help='指定するとカメラを起動する')
    parser.add_argument('--camera_select', type=int, default=0, help='カメラを指定した時、カメラデバイスのIDを指定')
    parser.add_argument('--models', default='models/weights_SSD300.hdf5', help='SSDモデルファイル')
    parser.add_argument('--labelfile', default="models/classname.txt", help='ラベル名称が書かれたファイル')
    parser.add_argument('--num_classes', type=int, default=21, help='SSDモデルの分類数')
    parser.add_argument('--thresh', type=float, default=0.7, help='物体識別の閾値')
    args = parser.parse_args()
    if not args.file and not args.camera:
        parser.print_help()
    else:
        main(args=args)
