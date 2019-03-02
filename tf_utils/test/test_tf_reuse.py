# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import unittest

import os
import random
import tensorflow as tf

from ml_tools.nets.mobilenet import mobilenet_v2
from ml_tools.tf_utils import get_tf_image_iterator_from_file_name, init_logger, SmartImageIterator, \
    ImageIteratorTestDemo, get_wsl_path
from pyxtools import byte_to_string, list_files

logger = logging.getLogger(__name__)
init_logger()


class _TestFixBug(object):
    model_name = "MobilenetV2"
    num_classes = 1001

    def __init__(self):
        self.pretrained_ckpt_file = get_wsl_path(
            "E:/frkhit/Download/AI/pre-trained-model/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt"
        )
        self.logger = logging.getLogger("_TestFixBug")
        self.image_size = 224
        self._queue_cache = None

    def _list_features_by_queue(self, image_iterator, feature_layer: str = "layer_18/output") -> (list, list):
        # todo reuse error
        # data set queue
        images, image_file_names = image_iterator.get_next()

        # load net
        images = tf.cast(images, tf.float32) / 128. - 1
        images.set_shape((None, None, None, 3))
        images = tf.image.resize_images(images, (self.image_size, self.image_size))

        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
            _, endpoints = mobilenet_v2.mobilenet(images, num_classes=self.num_classes)
        if feature_layer not in endpoints:
            self.logger.info("key of endpoints is {}".format(endpoints.keys()))
            raise ValueError("feature layer not in endpoints!")
        feature_extractor_layer = endpoints[feature_layer]

        saver = tf.train.Saver()

        # calc feature
        result_list = []
        file_list = []
        with tf.Session() as sess:
            saver.restore(sess, self.pretrained_ckpt_file)
            sess.run(image_iterator.initializer)
            while True:
                try:
                    _feature_list, _file_list = sess.run([feature_extractor_layer, image_file_names])
                    result_list.extend([feature for feature in _feature_list])
                    file_list.extend([byte_to_string(file_name) for file_name in _file_list])
                except tf.errors.OutOfRangeError:
                    break

        return result_list, file_list

    def _list_features_by_queue_correct(self, image_iterator, feature_layer: str = "layer_18/output") -> (list, list):
        if self._queue_cache and id(image_iterator) != id(self._queue_cache[0]):
            raise ValueError("image_iterator is different!")

        if self._queue_cache is None:
            # data set queue
            images, image_file_names = image_iterator.get_next()

            # load net
            images = tf.cast(images, tf.float32) / 128. - 1
            images.set_shape((None, None, None, 3))
            images = tf.image.resize_images(images, (self.image_size, self.image_size))

            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
                _, endpoints = mobilenet_v2.mobilenet(images, num_classes=self.num_classes)
            if feature_layer not in endpoints:
                self.logger.info("key of endpoints is {}".format(endpoints.keys()))
                raise ValueError("feature layer not in endpoints!")
            feature_extractor_layer = endpoints[feature_layer]
            saver = tf.train.Saver()
            self._queue_cache = (image_iterator, saver, feature_extractor_layer, image_file_names)

        x_image_iterator, saver, feature_extractor_layer, image_file_names = self._queue_cache
        # calc feature
        result_list = []
        file_list = []
        with tf.Session() as sess:
            saver.restore(sess, self.pretrained_ckpt_file)
            sess.run(x_image_iterator.initializer)
            while True:
                try:
                    _feature_list, _file_list = sess.run([feature_extractor_layer, image_file_names])
                    result_list.extend([feature for feature in _feature_list])
                    file_list.extend([byte_to_string(file_name) for file_name in _file_list])
                except tf.errors.OutOfRangeError:
                    break

        return result_list, file_list

    def test_list_features_by_queue_correct(self, image_iterator, feature_layer: str = "layer_18/output") -> (
            list, list):
        # error: reuse
        return self._list_features_by_queue_correct(image_iterator, feature_layer)

    def test_list_features_by_queue_v0(self, image_iterator, feature_layer: str = "layer_18/output") -> (list, list):
        # error: reuse
        return super(_TestFixBug, self)._list_features_by_queue(image_iterator, feature_layer)

    def test_list_features_by_queue_v1(self, image_iterator, feature_layer: str = "layer_18/output") -> (list, list):
        # error: reuse
        tf.reset_default_graph()
        with tf.Graph().as_default():
            return super(_TestFixBug, self)._list_features_by_queue(image_iterator, feature_layer)

    def test_list_features_by_queue_v2(self, image_iterator, feature_layer: str = "layer_18/output") -> (list, list):
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            images, image_file_names = image_iterator.get_next()

            # load net
            images = tf.cast(images, tf.float32) / 128. - 1
            images.set_shape((None, None, None, 3))
            images = tf.image.resize_images(images, (self.image_size, self.image_size))

            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
                _, endpoints = mobilenet_v2.mobilenet(images, num_classes=self.num_classes)
            if feature_layer not in endpoints:
                self.logger.info("key of endpoints is {}".format(endpoints.keys()))
                raise ValueError("feature layer not in endpoints!")
            feature_extractor_layer = endpoints[feature_layer]

            saver = tf.train.Saver()

            # calc feature
            result_list = []
            file_list = []
            with tf.Session(graph=graph) as sess:
                saver.restore(sess, self.pretrained_ckpt_file)
                sess.run(image_iterator.initializer)
                while True:
                    try:
                        _feature_list, _file_list = sess.run([feature_extractor_layer, image_file_names])
                        result_list.extend([feature for feature in _feature_list])
                        file_list.extend([byte_to_string(file_name) for file_name in _file_list])
                    except tf.errors.OutOfRangeError:
                        break

            return result_list, file_list

    def test_list_features_by_queue_v3(self, image_iterator, feature_layer: str = "layer_18/output") -> (list, list):
        if hasattr(self, "x") is False:
            self.x = None

        if self.x is None:
            # data set queue
            images, image_file_names = image_iterator.get_next()

            # load net
            images = tf.cast(images, tf.float32) / 128. - 1
            images.set_shape((None, None, None, 3))
            images = tf.image.resize_images(images, (self.image_size, self.image_size))

            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
                _, endpoints = mobilenet_v2.mobilenet(images, num_classes=self.num_classes)
            if feature_layer not in endpoints:
                self.logger.info("key of endpoints is {}".format(endpoints.keys()))
                raise ValueError("feature layer not in endpoints!")
            feature_extractor_layer = endpoints[feature_layer]
            saver = tf.train.Saver()
            self.x = (saver, feature_extractor_layer, image_file_names)

        saver, feature_extractor_layer, image_file_names = self.x

        # calc feature
        result_list = []
        file_list = []
        with tf.Session() as sess:
            saver.restore(sess, self.pretrained_ckpt_file)
            sess.run(image_iterator.initializer)
            while True:
                try:
                    _feature_list, _file_list = sess.run([feature_extractor_layer, image_file_names])
                    result_list.extend([feature for feature in _feature_list])
                    file_list.extend([byte_to_string(file_name) for file_name in _file_list])
                except tf.errors.OutOfRangeError:
                    break

        return result_list, file_list

    def test_list_features_by_queue_v4(self, image_iterator, feature_layer: str = "layer_18/output") -> (list, list):
        if hasattr(self, "y") is False:
            self.y = None

        if self.y is None:
            # data set queue
            images, image_file_names = image_iterator.get_next()

            # load net
            images = tf.cast(images, tf.float32) / 128. - 1
            images.set_shape((None, None, None, 3))
            images = tf.image.resize_images(images, (self.image_size, self.image_size))

            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
                _, endpoints = mobilenet_v2.mobilenet(images, num_classes=self.num_classes)
            if feature_layer not in endpoints:
                self.logger.info("key of endpoints is {}".format(endpoints.keys()))
                raise ValueError("feature layer not in endpoints!")
            feature_extractor_layer = endpoints[feature_layer]
            saver = tf.train.Saver()
            self.y = (saver, feature_extractor_layer, image_file_names, image_iterator)

        saver, feature_extractor_layer, image_file_names, x_image_iterator = self.y

        # calc feature
        result_list = []
        file_list = []
        with tf.Session() as sess:
            saver.restore(sess, self.pretrained_ckpt_file)
            sess.run(x_image_iterator.initializer)
            while True:
                try:
                    _feature_list, _file_list = sess.run([feature_extractor_layer, image_file_names])
                    result_list.extend([feature for feature in _feature_list])
                    file_list.extend([byte_to_string(file_name) for file_name in _file_list])
                except tf.errors.OutOfRangeError:
                    break

        return result_list, file_list


class TestTFReuseError(unittest.TestCase):
    def setUp(self):
        self.model = _TestFixBug()
        self._iterator = None

    def get_image_iterator(self, count: int = 1):
        image_list = [img_file for img_file in list_files(os.path.dirname(__file__)) if img_file.endswith(".jpg")]
        self.assertTrue(len(image_list) > 0)

        file_list = random.choices(image_list, k=count)
        self.assertTrue(len(file_list) == count)

        iterator = get_tf_image_iterator_from_file_name(image_list=file_list)

        return iterator, file_list

    def tearDown(self):
        pass

    def check_method(self, test_method):
        for count in range(1, 3):
            image_iterator, image_list = self.get_image_iterator(count=count)

            try:
                feature_list, file_list = test_method(image_iterator)
                success = True
                self.assertEqual(len(feature_list), count)
                self.assertEqual(len(file_list), count)

                for i, image_file in enumerate(image_list):
                    self.assertEqual(image_file, file_list[i])

            except Exception as e:
                logger.error(e, exc_info=True)
                success = False

            self.assertTrue(success)

    def check_method_with_same_iterator(self, test_method):
        all_image_list = [img_file for img_file in list_files(os.path.dirname(__file__)) if
                          img_file.endswith(".jpg")] * 2

        smart_iter = SmartImageIterator(tf_decoder_func=ImageIteratorTestDemo.tf_decode_jpg_with_size)
        for count in range(1, len(all_image_list) + 1):
            new_file_list = random.choices(all_image_list, k=count)
            self.assertTrue(len(new_file_list) == count)
            smart_iter.set_file_list(new_file_list)

            image_iterator = smart_iter.get_tf_image_iterator()

            try:
                feature_list, file_list = test_method(image_iterator)
                success = True
                self.assertEqual(len(feature_list), count)
                self.assertEqual(len(file_list), count)

                for i, image_file in enumerate(new_file_list):
                    self.assertEqual(image_file, file_list[i])

            except Exception as e:
                logger.error(e, exc_info=True)
                success = False

            self.assertTrue(success)

    def test_v0(self):
        # bug: fail
        self.check_method(self.model.test_list_features_by_queue_v0)

    def test_v1(self):
        # fix bug use method v1: fail
        self.check_method(self.model.test_list_features_by_queue_v1)

    def test_v2(self):
        # fix bug use method v2: fail
        self.check_method(self.model.test_list_features_by_queue_v2)

    def test_v3(self):
        # fix bug use method v2: fail
        self.check_method(self.model.test_list_features_by_queue_v3)

    def test_tf_reuse(self):
        def tf_reuse():
            # success
            bob = tf.Variable(name="x", initial_value=0.1, dtype=tf.float32)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                logger.info("bob is {}".format(bob.eval()))

        success = True
        try:
            for i in range(3):
                tf_reuse()
        except Exception as e:
            logger.error(e, exc_info=True)
            success = False

        self.assertTrue(success)

    def test_tf_reuse_different_graph(self):
        def run():
            tf.reset_default_graph()
            with tf.Graph().as_default() as graph:
                bob = tf.Variable(name="x", initial_value=0.1, dtype=tf.float32)
                with tf.Session(graph=graph) as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    logger.info("bob is {}".format(bob.eval()))

        success = True
        try:
            for i in range(3):
                run()
        except Exception as e:
            logger.error(e, exc_info=True)
            success = False

        self.assertTrue(success)

    def test_i_v0(self):
        self.check_method_with_same_iterator(self.model.test_list_features_by_queue_v0)

    def test_i_v1(self):
        self.check_method_with_same_iterator(self.model.test_list_features_by_queue_v1)

    def test_i_v2(self):
        self.check_method_with_same_iterator(self.model.test_list_features_by_queue_v2)

    def test_i_v3(self):
        self.check_method_with_same_iterator(self.model.test_list_features_by_queue_v3)

    def test_i_v4(self):
        self.check_method_with_same_iterator(self.model.test_list_features_by_queue_v4)

    def test_x_v0(self):
        self.check_method_with_same_iterator(self.model.test_list_features_by_queue_correct)
