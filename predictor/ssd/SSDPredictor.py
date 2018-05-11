#-*- coding: utf-8 -*-

# Keras SSD Predictor

import numpy as np

from ..base import PredictorBase
from ..vendors import ssd, ssd_utils
from ssd import SSD300 as SSD
from ssd_utils import BBoxUtility
from keras.applications.imagenet_utils import preprocess_input

class SSDPrecitor(PredictorBase):
    """
    SSDの識別器
    Arguments:
        modelfile: モデルファイルパス
        shape: SSD識別器に入力する際のモデルサイズ. デフォルトは (300, 300, 3)
        num_classes: モデルの分類数. デフォルトは 21
    """

    def __init__(self, modelfile, shape=(300, 300, 3), num_classes=21):

        self.input_shape = shape
        self.num_classes = num_classes

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
        pass

    def _preprocess(self, src):
        """
        入力された画像に対して前処理を行う
        """
        pass

    def _postprocess(self, src):
        """
        識別処理後の後処理
        """
        pass
