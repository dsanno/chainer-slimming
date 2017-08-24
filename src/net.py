import math
import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainer import function
from chainer import link
from chainer.utils import array
from chainer.utils import type_check

class BatchConv2D(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize, stride=1, pad=0, activation=F.relu):
        super(BatchConv2D, self).__init__(
            conv=L.Convolution2D(ch_in, ch_out, ksize, stride, pad),
            bn=L.BatchNormalization(ch_out),
        )
        self.activation=activation

    def __call__(self, x):
        h = self.bn(self.conv(x))
        if self.activation is None:
            return h
        return self.activation(h)

class VGG(chainer.Chain):
    def __init__(self, output_sizes):
        super(VGG, self).__init__()
        with self.init_scope():
            input_size = 3
            for name in VGG_LAYERS:
                if name =='fc':
                    break
                output_size = output_sizes[name]
                setattr(self, name, BatchConv2D(input_size, output_size, 3,
                    stride=1, pad=1))
                input_size = output_size
            self.fc = L.Linear(input_size, 10)

    def __call__(self, x):
        h = self.bconv1_1(x)
        h = self.bconv1_2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.bconv2_1(h)
        h = self.bconv2_2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.bconv3_1(h)
        h = self.bconv3_2(h)
        h = self.bconv3_3(h)
        h = self.bconv3_4(h)
        h = F.max_pooling_2d(h, 2)
        h = self.bconv4_1(h)
        h = self.bconv4_2(h)
        h = self.bconv4_3(h)
        h = self.bconv4_4(h)
        h = F.max_pooling_2d(h, 2)
        h = self.bconv5_1(h)
        h = self.bconv5_2(h)
        h = self.bconv5_3(h)
        h = self.bconv5_4(h)
        h = F.average_pooling_2d(h, 2)
        h = self.fc(h)
        return h

VGG_LAYERS = [
    'bconv1_1',
    'bconv1_2',
    'bconv2_1',
    'bconv2_2',
    'bconv3_1',
    'bconv3_2',
    'bconv3_3',
    'bconv3_4',
    'bconv4_1',
    'bconv4_2',
    'bconv4_3',
    'bconv4_4',
    'bconv5_1',
    'bconv5_2',
    'bconv5_3',
    'bconv5_4',
    'fc',
]
