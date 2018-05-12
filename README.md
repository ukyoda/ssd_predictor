# SSD 識別器を動かす

## Overview

[Keras版SSD](https://github.com/rykov8/ssd_keras)で
物体識別を簡単に試すことができるラッパークラスです。

## QuickStart

下記コマンドソースコードをクローンし、ssd_demo.pyを実行してください

```bash
$ git clone --recursive https://github.com/ukyoda/ssd-predictor.git
```

デモプログラムを実行する上で必要なのんは下記のオプションです

```bash
$ python ssd_demo.py -h
Using TensorFlow backend.
usage: ssd_demo.py [-h] [--file FILE] [--camera]
                   [--camera_select CAMERA_SELECT]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           動画ファイルを指定(カメラモードの時は見指定でOK)
  --camera              指定するとカメラを起動する
  --camera_select CAMERA_SELECT
                        カメラを指定した時、カメラデバイスのIDを指定
```

## 動作要件

* Python3推奨. Python2では動かないかも
* 本家SSDが動く環境
* Kerasのみ、2.0で動くように調整しています(本家をクローンして必要な修正を施したもの)

## QuickStart

## TODOリスト(やるかは不明)

* SSDを学習するプログラムを作る
*
