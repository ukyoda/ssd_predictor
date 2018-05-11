#-*- coding: utf-8 -*-

class PredictorBase:

    def predict(self, src):
        """Predictor base class
        :param src: input data
        :return: prediction result
        """
        raise NotImplementedError()
        
