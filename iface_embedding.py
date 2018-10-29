#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import numpy as np
import mxnet as mx
import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = '0'
from sklearn.preprocessing import normalize
from .iface_config import IFaceConfig


class IFaceEmbedding(object):
    def __init__(self, image_size=(100,100)):
        self.config = IFaceConfig()
        self.image_size = image_size
        self.init_model(self.config.MODEL)

    def init_model(self, model_str, layer="fc1"):
        if self.config.USE_CUDA:
            ctx = mx.gpu(0)
        else:
            ctx = mx.cpu(0)

        _vec = model_str.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer+'_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, self.image_size[0],
                                          self.image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def get_feature(self, face_region):
        face_region = cv2.resize(face_region, self.image_size)
        face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        face_region = np.transpose(face_region, (2,0,1))
        input_blob = np.expand_dims(face_region, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = normalize(embedding).flatten()
        return embedding

    def test_image(self):
        imgs = [cv2.imread(_img_path) for _img_path in self.config.TEST_IMAGES]
        embeddings = [self.get_feature(img) for img in imgs]
        e1, e2 = embeddings
        dist = np.sum(np.square(e1-e2))
        return e1, e2, dist


def main():
    iface = IFaceEmbedding()
    imgs = [cv2.imread(_img_path) for _img_path in iface.config.TEST_IMAGES]
    embeddings = [iface.get_feature(img) for img in imgs]
    e1, e2 = embeddings
    dist = np.sum(np.square(e1-e2))
    print(dist)


if __name__ == '__main__':
    main()
