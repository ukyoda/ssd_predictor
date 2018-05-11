#-*- coding: utf-8 -*-

import sys
import os

def addpath(dir):
    if dir not in sys.path:
        sys.path.insert(0, dir)

def poppath():
    sys.path = sys.path[1:]

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))

# SSDをインポートする
# ------------------------------------------------------------------------------
addpath(os.path.join(ROOT_DIR, 'vendors/ssd_keras'))
import ssd
from ssd_utills import BBoxUtility
poppath()
