#-*- coding: utf-8 -*-

# 各種ユーティリティ関数
# ==============================================================================

import numpy as np
import cv2

def create_class_colors(num_classes):
    """
    クラスの色を作成する(SSD本家の色をつける処理を移植)
    """
    colors = []
    for i in range(num_classes):
        # 本家では255にしているけど、色相ってOpenCVでは180までだったはずなので180にすることにする
        hue = 180 * i / num_classes
        hsv_color = np.zeros((1, 1, 3)).astype("uint8")
        hsv_color[0][0][0] = hue
        hsv_color[0][0][1] = 128
        hsv_color[0][0][2] = 255
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        color = (int(bgr_color[0, 0, 0]), int(bgr_color[0, 0, 1]), int(bgr_color[0, 0, 2]))
        colors.append(color)
    return colors

def create_classnames(num_classes=0, filepath=None):
    """
    クラス名称を作成する
    **クラスファイルのテキストファイルを指定したらそれを使う**
    """
    if filepath is None:
        return ["CLASS_{:d}".format(x) for x in range(num_classes)]
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(line)

    return labels
