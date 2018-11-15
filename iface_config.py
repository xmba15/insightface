#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import torch


_DIRECTORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))
_MODEL_PATH = os.path.join(_DIRECTORY_ROOT, "models")
_DATA_PATH = os.path.join(_DIRECTORY_ROOT, "data")
_MODEL_R50 = os.path.join(_MODEL_PATH, "model-r50-am-lfw/model,00")
_MODEL_R100 = os.path.join(_MODEL_PATH, "model-r100-ii/model,00")
_TEST_IMAGES = [os.path.join(_DATA_PATH, img) for img in ["nakama1.jpg", "nakama2.jpg"]]


class IFaceConfig(object):
    def __init__(self):
        self.MODEL_R50 = _MODEL_R50
        self.MODEL_R100 = _MODEL_R100
        self.USE_CUDA = torch.cuda.is_available()
        self.TEST_IMAGES = _TEST_IMAGES

    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


def main():
    config = IFaceConfig()
    config.display()


if __name__ == '__main__':
    main()
