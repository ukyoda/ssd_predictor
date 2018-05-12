# SSD物体検出ライブラリ

## Overview

[Keras版SSD](https://github.com/rykov8/ssd_keras)を使いやすくしたラッパーライブラリ

## 動作要件

* Python3推奨
* Tensorflow (動作確認はCPU版で実施)
* OpenCV3
* Keras2

## サンプル実行方法

### 準備

下記コマンドソースコードをクローンしてください(--recursiveオプションをつけること)

```bash
$ git clone --recursive https://github.com/ukyoda/ssd-predictor.git
```

* モデルファイルを用意し、modelsディレクトリに設置してください([こことかからDLしてください](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA))
* モデルファイルに合わせてmodels/classname.txtを書き換えてください

### 実行方法

```bash
$ python ssd_demo.py -h
Using TensorFlow backend.
usage: ssd_demo.py [-h] [--file FILE] [--camera]
                   [--camera_select CAMERA_SELECT] [--models MODELS]
                   [--num_classes NUM_CLASSES] [--thresh THRESH]
                   [--labelfile LABELFILE]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           動画ファイルを指定(カメラモードの時は見指定でOK)
  --camera              指定するとカメラを起動する
  --camera_select CAMERA_SELECT
                        カメラを指定した時、カメラデバイスのIDを指定
  --models MODELS       SSDモデルファイル
  --num_classes NUM_CLASSES
                        SSDモデルの分類数
  --thresh THRESH       物体識別の閾値
  --labelfile LABELFILE
                        ラベル名称が書かれたファイル
```

カメラで実行する方法

```bash
$ python ssd_demo.py --camera
```

## TODOリスト(やるかは不明)

* SSDを学習するプログラムを作る
* SSD以外のモデルも試してみる

## 利用規約について

* ライセンス、利用規約などは本家のSSDに準拠します (MITライセンス)
* 本ライブラリは予告なく終了することがあります
