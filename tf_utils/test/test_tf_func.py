# -*- coding:utf-8 -*-
from __future__ import absolute_import

import unittest

import numpy as np
import tensorflow as tf

from ml_tools.tf_utils import init_logger, whale_siamese_image_mean_np, whale_siamese_image_mean_tf

init_logger(None)


class TestTFFunc(unittest.TestCase):
    def testImageMean(self):
        def tf_process(arr_list: list, arr_shape) -> list:
            with tf.Graph().as_default() as graph:
                image = tf.placeholder(dtype=tf.float32, shape=arr_shape, name='embedding')
                np_image = whale_siamese_image_mean_tf(image)

                array_list = []
                with tf.Session(graph=graph) as sess:
                    for arr in arr_list:
                        array_list.append(sess.run(np_image, feed_dict={image: arr}))

                return array_list

        test_shape = (384, 384, 1)
        test_array_list = [np.random.random(test_shape).astype(np.float32) + 10 - i for i in range(10)]

        np_std_list = [whale_siamese_image_mean_np(img) for img in test_array_list]
        tf_std_list = tf_process(test_array_list, test_shape)

        for index in range(len(test_array_list)):
            self.assertTrue((abs(np_std_list[index] - tf_std_list[index]) < 1e-4).all())
