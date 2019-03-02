# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import unittest

import os
import random
import shutil
import tensorflow as tf
from tensorflow.python.keras import Input, backend as K, regularizers
from tensorflow.python.keras.layers import Activation, Add, BatchNormalization, Conv2D, GlobalMaxPooling2D, \
    MaxPooling2D, Lambda, Concatenate, Reshape, Flatten, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

from ml_tools.tf_utils import init_logger, DatasetUtils, Params, keras_convert_model_to_estimator_ckpt, MnistKerasDemo
from ml_tools.tf_utils.test.utils import compare_two_dir

init_logger(None)


class TestKerasUtils(unittest.TestCase):
    def setUp(self):
        self.dir_1 = "./dir1"
        self.dir_2 = "./dir2"
        self._two_dir_is_same = False
        if not os.path.exists(self.dir_1):
            os.mkdir(self.dir_1)

        if not os.path.exists(self.dir_2):
            os.mkdir(self.dir_2)

        # mnist demo
        self.mnist_keras_model = "./k.model"
        self.mnist_tf_dir = "./tf_dir"

    def tearDown(self):
        if self._two_dir_is_same:
            if os.path.exists(self.dir_1):
                shutil.rmtree(self.dir_1)

            if os.path.exists(self.dir_2):
                shutil.rmtree(self.dir_2)

    def _mnist_demo(self, run_count: int):
        mnist = MnistKerasDemo()
        mnist.keras_model_file = self.mnist_keras_model
        mnist.tf_dir = self.mnist_tf_dir

        all_true = True
        x_train = mnist.x_train.copy()
        y_train = mnist.y_train.copy()
        _train_length = len(mnist.x_train) // 128
        for i in range(run_count):
            _count = random.choice([ii for ii in range(_train_length)])
            mnist.x_train = x_train[_count * 128:(_count + 1) * 128]
            mnist.y_train = y_train[_count * 128:(_count + 1) * 128]
            mnist.keras_train()

            is_equal = mnist.print_tf_estimator() == mnist.print_keras_model()
            if all_true and not is_equal:
                all_true = False

            self.assertTrue(is_equal)

        if all_true:
            if os.path.exists(self.mnist_tf_dir):
                shutil.rmtree(self.mnist_tf_dir)

            if os.path.exists(self.mnist_keras_model):
                os.remove(self.mnist_keras_model)

    def testMnistDemoV0(self):
        """
            Pass
        Returns:

        """
        self._mnist_demo(run_count=10)

    def testMnistDemoV1(self):
        """
            todo Fail
        Returns:

        """
        self._mnist_demo(run_count=1)

    def testKerasToTF(self):
        """
            Warnings: test not pass
        Returns:

        """

        def get_bone_net():
            """ 返回func """

            def sub_block(x, n_filter, **kwargs):
                x = BatchNormalization()(x)
                y = x
                y = Conv2D(n_filter, (1, 1), activation='relu', **kwargs)(
                    y)  # Reduce the number of features to 'filter'
                y = BatchNormalization()(y)
                y = Conv2D(n_filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
                y = BatchNormalization()(y)
                y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(
                    y)  # no activation # Restore the number of original features
                y = Add()([x, y])  # Add the bypass connection
                y = Activation('relu')(y)
                return y

            def get_model_net(is_training: bool, img_shape: tuple, ):
                if is_training:
                    regul = regularizers.l2(0.0002)
                else:
                    regul = regularizers.l2(0)
                kwargs = {'padding': 'same', 'kernel_regularizer': regul}

                inp = Input(shape=img_shape)  # 384x384x1
                x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

                x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
                for _ in range(2):
                    x = BatchNormalization()(x)
                    x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

                x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
                x = BatchNormalization()(x)
                x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
                for _ in range(4):
                    x = sub_block(x, 64, **kwargs)

                x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
                x = BatchNormalization()(x)
                x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
                for _ in range(4):
                    x = sub_block(x, 64, **kwargs)

                x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
                x = BatchNormalization()(x)
                x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
                for _ in range(4):
                    x = sub_block(x, 96, **kwargs)

                x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
                x = BatchNormalization()(x)
                x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
                for _ in range(4):
                    x = sub_block(x, 128, **kwargs)

                x = GlobalMaxPooling2D()(x)  # 512
                return Model(inputs=inp, outputs=x, name="xxx")

            return get_model_net

        def model_func(mode, params):
            get_model_net = get_bone_net()
            img_shape = (params.image_size, params.image_size, 1)
            if mode == tf.estimator.ModeKeys.TRAIN:
                branch_model = get_model_net(is_training=True, img_shape=img_shape)
            else:
                branch_model = get_model_net(is_training=False, img_shape=img_shape)

            ############
            # HEAD MODEL
            ############
            mid = 32
            xa_inp = Input(shape=branch_model.output_shape[1:])
            xb_inp = Input(shape=branch_model.output_shape[1:])
            x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
            x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
            x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
            x4 = Lambda(lambda x: K.square(x))(x3)
            x = Concatenate()([x1, x2, x3, x4])
            x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

            # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
            x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
            x = Reshape((branch_model.output_shape[1], mid, 1))(x)
            x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
            x = Flatten(name='flatten')(x)

            # Weighted sum implemented as a Dense layer.
            x = Dense(1, use_bias=True, activation='sigmoid', name='weighted-average')(x)
            head_model = Model([xa_inp, xb_inp], x, name='head')

            ########################
            # SIAMESE NEURAL NETWORK
            ########################
            # Complete model is constructed by calling the branch model on each input image,
            # and then the head model on the resulting 512-vectors.
            img_a = Input(shape=img_shape, name="xa")
            img_b = Input(shape=img_shape, name="xb")
            xa = branch_model(img_a)
            xb = branch_model(img_b)
            x = head_model([xa, xb])
            model = Model([img_a, img_b], x)
            model.compile(Adam(lr=1e-4), loss='binary_crossentropy',
                          metrics=['binary_crossentropy', 'acc'])
            return model

        def tf_keras(log_dir: str):
            keras_convert_model_to_estimator_ckpt(keras_model_path=model_file, log_dir=log_dir)

        def tf_keras_v1(log_dir: str):
            base_model = model_func(mode=DatasetUtils.SPLIT_TRAIN, params=Params({"image_size": 384}))
            keras_convert_model_to_estimator_ckpt(keras_model_path=model_file, keras_model=base_model, log_dir=log_dir)

        model_file = "/mnt/e/frkhit/Download/AI/data-set/kaggle/whale/mpiotte-standard.model"
        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            logging.warning("model_file {} not exists!".format(model_file))
            return

        tf_keras(self.dir_1)
        tf_keras_v1(self.dir_2)

        self._two_dir_is_same = compare_two_dir(self.dir_1, self.dir_2)
        self.assertTrue(self._two_dir_is_same)
