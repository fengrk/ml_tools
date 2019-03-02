# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import time
import unittest

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ml_tools.tf_utils import init_logger, estimator_iter_process, AbstractEstimator, DatasetUtils

init_logger()


class E(AbstractEstimator):
    """
        todo: trying to make estimator restore from pb file
    """

    def __init__(self, model_name, train_ckpt_dir, pretrained_ckpt_file=None):
        super(E, self).__init__(model_name, train_ckpt_dir, pretrained_ckpt_file)
        self._features_x_key = "x"
        self._features_y_key = "y"

    def get_dataset_func(self, split_name, num_epochs=1, shuffle=True, batch_size=64, num_parallel_calls=2,
                         prefetch_size=2, shuffle_size=4):
        _shuffle = False
        _num_epochs = 1

        if split_name == DatasetUtils.SPLIT_TRAIN:
            train_data = np.random.rand(num_epochs * batch_size * 12800) * 10.0
            train_data = train_data.reshape((train_data.shape[0], 1))
            train_labels = 0.5 * train_data + 0.2 + np.random.normal(loc=0, scale=0.1, size=train_data.shape)
            _shuffle, _num_epochs = True, num_epochs
        else:
            train_data = np.arange(0, 10) * 1.0
            train_data = train_data.reshape((train_data.shape[0], 1))
            train_labels = 0.5 * train_data + 0.2

        # logging.info("shape of x is {}, shape of y is {}".format(train_data.shape, train_labels.shape))

        return tf.estimator.inputs.numpy_input_fn(
            x={self._features_x_key: train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=_num_epochs,
            shuffle=_shuffle,
            num_threads=1
        )

    def model_fun(self, ):
        """ 返回func """

        def get_model_net(scope_name, features):
            inputs = tf.reshape(features[self._features_x_key], [-1, 1])

            with tf.variable_scope(scope_name, 'simple_fc', [inputs]) as sc:
                end_points_collection = sc.original_name_scope + '_end_points'

                with slim.arg_scope([slim.fully_connected], outputs_collections=[end_points_collection]):
                    net = slim.fully_connected(inputs, num_outputs=1, scope="fc",
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               normalizer_params=None,
                                               weights_initializer=None,
                                               weights_regularizer=None,
                                               biases_initializer=None,
                                               biases_regularizer=None, )

                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                    return net, end_points

        def model_func(features, labels, mode, params):
            # logging.info("shape of feature is {}, shape of label is {}".format(
            #     features[self._features_x_key].shape, labels.shape))

            net, end_points = get_model_net(scope_name=self.model_name, features=features)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {self._features_y_key: net}
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            loss = tf.square(tf.subtract(labels, net))
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.get_learning_rate())
                global_step = tf.train.get_global_step()
                train_op = optimizer.minimize(loss, global_step=global_step, var_list=None)

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

            eval_metric_ops = {'acc': tf.metrics.mean_squared_error(labels=labels, predictions=net)}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

        return model_func


class TestProjectDemo(unittest.TestCase):
    def testEstimatorIterProcess(self):
        def loop_process(total, num, time_sleep: float = 1):
            time.sleep(time_sleep)
            logging.info("total is {}, num is {}".format(total, num))

        def end_process_func(time_sleep: float = 1):
            time.sleep(time_sleep)
            logging.info("end_process_func")

        estimator_iter_process(loop_process, iter_stop_time=1e12, loop_process_min_epoch=1,
                               end_process_func=end_process_func, loop_process_max_epoch=10,
                               ignore_error_in_loop_process=True)

    def testEstimatorPbFile(self):
        estimator = E(model_name="E", train_ckpt_dir="./y", pretrained_ckpt_file=None)

        logging.info("acc is {}".format(estimator.evaluate(num_epochs=1, batch_size=1)))
        estimator.train(batch_size=1, num_epochs=10)

        logging.info("acc is {}".format(estimator.evaluate(num_epochs=1, batch_size=1)))
