#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import time

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "ResNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"
]


class ResNet():
    def __init__(self, layers=50):
        self.layers = layers

    def net(self, input, class_dim=1000, data_format="NCHW"):
        layers = self.layers
        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1",
            data_format=data_format)
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max',
            data_format=data_format)
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name=conv_name,
                        data_format=data_format)

            pool = fluid.layers.pool2d(
                input=conv, pool_type='avg', global_pooling=True, data_format=data_format)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=class_dim,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)))
        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block == i == 0,
                        name=conv_name,
                        data_format=data_format)

            pool = fluid.layers.pool2d(
                input=conv, pool_type='avg', global_pooling=True, data_format=data_format)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=class_dim,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None,
                      data_format='NCHW'):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1',
            data_format=data_format)

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)

    def shortcut(self, input, ch_out, stride, is_first, name, data_format):
        if data_format == 'NCHW':
            ch_in = input.shape[1]
        else:
            ch_in = input.shape[-1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name, data_format=data_format)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name, data_format):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a",
            data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b",
            data_format=data_format)
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format)

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=False,
            name=name + "_branch1",
            data_format=data_format)

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")

    def basic_block(self, input, num_filters, stride, is_first, name, data_format):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a",
            data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b",
            data_format=data_format)
        short = self.shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1", data_format=data_format)
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')


def ResNet18():
    model = ResNet(layers=18)
    return model


def ResNet34():
    model = ResNet(layers=34)
    return model


def ResNet50():
    model = ResNet(layers=50)
    return model


def ResNet101():
    model = ResNet(layers=101)
    return model


def ResNet152():
    model = ResNet(layers=152)
    return model


def get_random_images_and_labels(batch_size):
    image = np.random.randn(batch_size, 3, 224, 224).astype('float32')
    label = np.random.randint(0, 1000, [batch_size, 1]).astype('int64')
    return image, label


def batch_generator_creator(batch_size):
    BATCH_NUM = 100
    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = get_random_images_and_labels(batch_size)
            yield batch_image, batch_label

    return __reader__



def train():

    paddle.enable_static()

    main_prog = fluid.Program()
    start_prog = fluid.Program()
    with fluid.program_guard(main_prog, start_prog):
        # placeholder
        img = fluid.data(shape=[None, 3, 224, 224], name='img')
        label = fluid.data(shape=[None, 1],dtype='int64', name='label')
        # build model
        model = ResNet50()
        out = model.net(img, class_dim=1000) # align with dygraph
        pred = fluid.layers.softmax(out)
        # optimizer
        optimizer = fluid.optimizer.Adam(learning_rate=0.01)
        cost = fluid.layers.cross_entropy(input=pred, label=label)
        loss = fluid.layers.mean(cost)
        optimizer.minimize(loss)
        
        # run program
        exe = fluid.Executor(fluid.CUDAPlace(0))
        exe.run(start_prog)
        main_prog = fluid.CompiledProgram(main_prog)

        for batch_size in [1, 1, 4, 16, 32]:
            # dataloader
            # loader = fluid.io.DataLoader.from_generator(feed_list=[img, label], capacity=16, iterable=True)
            # loader.set_batch_generator(batch_generator_creator(batch_size), places=fluid.cuda_places())

            x = np.random.randn(batch_size, 3, 224, 224).astype('float32')
            gt_label = np.random.randint(0, 1000, [batch_size, 1]).astype('int64')
            # for data in loader():
                # pred_out = exe.run(main_prog, feed=data, fetch_list=[pred])
            start = time.time()
            for i in range(100):
                pred_out = exe.run(main_prog, feed={'img': x, 'label':gt_label}, fetch_list=[pred])
            end = time.time()
            print("batch_size: {}, cost: {} ms.".format(batch_size, (end - start)*10))

if __name__ == '__main__':
    train()