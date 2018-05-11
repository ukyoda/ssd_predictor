#-*- coding: utf-8 -*-

# Keras SSD Predictor

import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from ..base import PredictorBase
from ..vendors import ssd, ssd_utils
from ssd import SSD300 as SSD
from ssd_utils import BBoxUtility

class SSDPrecitor(PredictorBase):
    """
    SSDの識別器
    Arguments:
        modelfile: モデルファイルパス
        shape: SSD識別器に入力する際のモデルサイズ(width, height, channels). デフォルトは (300, 300, 3)
        num_classes: モデルの分類数. デフォルトは 21
        conf_thresh: 検出結果の閾値
    """

    def __init__(self, modelfile, shape=(300, 300, 3), num_classes=21, conf_thresh=0.6):

        self.input_shape = shape
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh

        # モデル作成
        model = SSD(shape, num_classes=num_classes)
        model.load_weights(modelfile)
        self.model = model

        # バウンディングボックス作成ユーティリティ
        self.bbox_util = BBoxUtility(self.num_classes)

    def predict(self, src):
        """
        SSDにより、入力画像からオブジェクトを識別する
        """
        height, width, channels = src.shape
        # 前処理
        x = self._preprocess(src)
        # 推論
        y = self.model.predict(x)
        # 後処理
        results = self._decodebox(y)
        # 出力
        if results.shape[0] > 0:
            results[:, 2] = results[:, 2] * width
            results[:, 3] = results[:, 3] * height
            results[:, 4] = results[:, 4] * width
            results[:, 5] = results[:, 5] * height
            return [(int(x[0]), x[1], x[2:6]) for x in results]

        return []

    def _preprocess(self, src):
        """
        入力された画像に対して前処理を行う
        """
        im_size = (self.input_shape[0], self.input_shape[1])
        resized = cv2.resize(src, im_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inputs = [image.img_to_array(rgb)]
        return preprocess_input(np.array(inputs))


    def _decodebox(self, preds):
        """
        識別処理後の後処理
        """
        conf_thresh=self.conf_thresh

        # ボックス抽出 **これで下記の情報が取得する**
        #   0: label
        #   1: conf
        # 2~5: bbox(xmin, ymin, xmax, ymax)
        box_results = self.bbox_util.detection_out(preds)
        result = np.array([])
        if len(box_results) > 0 and len(box_results[0]) > 0:
            box_result = box_results[0]
            # スコアが閾値以上のデータのインデックスを取り出す
            top_indices = np.where(box_result[:, 1] > self.conf_thresh)[0]
            result = box_result[top_indices]
            return result
        else:
            return result
