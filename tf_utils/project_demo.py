# -*- coding:utf-8 -*-
from __future__ import absolute_import

import pickle

import numpy as np
import os
import shutil
import tensorflow as tf

from pyxtools import byte_to_string

try:
    from pymltools.tf_utils import *
except ImportError:
    from pymltools.pymltools.tf_utils import *

from . import face_losses


class TripletLossModelMnist(AbstractEstimator):
    def __init__(self, train_ckpt_dir, pretrained_ckpt_file: str = None):
        super(TripletLossModelMnist, self).__init__(
            model_name="TripletLoss",
            train_ckpt_dir=train_ckpt_dir,
            pretrained_ckpt_file=pretrained_ckpt_file
        )

        self._features_images_key = "images"
        self._features_filename_key = "filenames"
        self._features_embedding_key = "embeddings"

        self.train_loss_hook = LossStepHookForTrain(log_after_run=True, log_after_end=False, show_log_per_steps=20)
        self.params = Params({
            "num_channels": 1,
            "num_filter": 32,
            "margin": 1.0,
            "dimension": 64,
            "image_size": 28,
            "online_batch_count": 32,
        })
        self.learning_rate = 64e-5
        self.learning_rate_decay_epoch_num = 2

        # data set
        pkl_file = os.path.join(self.TRAIN_CKPT_DIR, "mnist.pkl")
        if not os.path.exists(pkl_file):
            self.mnist_dict = {"train": {}, "test": {}}
            mnist_dataset = get_mini_train_set(validation_size=0)
            self.mnist_dict["train"]["images"] = mnist_dataset.train.images
            self.mnist_dict["train"]["labels"] = mnist_dataset.train.labels

            self.mnist_dict["test"]["images"] = mnist_dataset.test.images
            self.mnist_dict["test"]["labels"] = mnist_dataset.test.labels

            with open(pkl_file, "wb", ) as f:
                pickle.dump(self.mnist_dict, f)
        else:
            with open(pkl_file, "rb", ) as f:
                self.mnist_dict = pickle.load(f)
        self.train_data_size = len(self.mnist_dict["train"]["labels"])

    def model_fun(self, ):
        """ 返回func """

        def get_model_net(scope_name, features, params, is_training, ):
            inputs = tf.reshape(features[self._features_images_key],
                                [-1, params.image_size, params.image_size, params.num_channels])

            with tf.variable_scope(scope_name, 'triplet_loss', [inputs]) as sc:
                out = inputs
                num_filter = params.num_filter
                channels = [num_filter, num_filter * 2]
                for i, c in enumerate(channels):
                    with tf.variable_scope('block_{}'.format(i + 1)):
                        out = tf.layers.conv2d(out, c, 3, padding='same')
                        out = tf.layers.batch_normalization(out, momentum=0.9, training=is_training)
                        out = tf.nn.relu(out)
                        out = tf.layers.max_pooling2d(out, 2, 2)

                assert out.shape[1:] == [7, 7, num_filter * 2]

                out = tf.reshape(out, [-1, 7 * 7 * num_filter * 2])
                with tf.variable_scope('fc_1'):
                    out = tf.layers.dense(out, params.dimension)

                return out, {}

        def model_func(features, labels, mode, params):
            """Model function for tf.estimator

                Args:
                    features: input batch of images
                    labels: labels of the images
                    mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
                    params: contains hyperparameters of the model (ex: `params.learning_rate`)

                Returns:
                    model_spec: tf.estimator.EstimatorSpec object
                """
            if mode == tf.estimator.ModeKeys.TRAIN:
                embeddings, end_points = get_model_net(scope_name=self.model_name, features=features, params=params,
                                                       is_training=True)
                TensorboardUtils.variable_list_summaries()
            else:
                embeddings, end_points = get_model_net(scope_name=self.model_name, features=features, params=params,
                                                       is_training=False)

            embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
            tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

            # axis = 0: means normalize each dim by info from this batch
            # axis = 1: means normalize x each dim only by x info
            # normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=0)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    self._features_embedding_key: embeddings,
                    self._features_filename_key: features[self._features_filename_key]
                }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            labels = tf.cast(labels, tf.int64)

            # Define triplet loss
            loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin, squared=False)

            with tf.variable_scope("metrics"):
                eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

            # Summaries for training
            tf.summary.scalar('loss', loss)

            # Define training step that minimizes the loss with the Adam optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.get_learning_rate())
            global_step = tf.train.get_global_step()

            # batch norm need this code:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step, var_list=None)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        return model_func

    def get_file_list(self, split_name: str) -> list:

        if split_name == DatasetUtils.SPLIT_TRAIN:
            image_list = self.mnist_dict["train"]["images"]
            labels_list = self.mnist_dict["train"]["labels"]
        else:
            image_list = self.mnist_dict["test"]["images"]
            labels_list = self.mnist_dict["test"]["labels"]

        return [image_list, labels_list]

    def list_features(self, file_list: list = None, mode: ProcessMode = None) -> (list, list):
        if file_list is None and mode is not None:
            if mode == ProcessMode.train:
                input_list = self.get_file_list(split_name=DatasetUtils.SPLIT_TRAIN)
            else:
                input_list = self.get_file_list(split_name=DatasetUtils.SPLIT_PREDICT)
        else:
            input_list = file_list

        input_fn = self.get_dataset_func(
            split_name=DatasetUtils.SPLIT_PREDICT, input_list=input_list,
            num_epochs=1, shuffle=False, batch_size=self.batch_size, num_parallel_calls=2, prefetch_size=2,
        )

        # predict
        self.logger.info("trying to list feature for {} file...".format(len(input_list)))
        classifier = self.get_classifier()
        result_list = classifier.predict(input_fn=input_fn)

        # parse feature
        feature_list = []
        feature_file_list = []
        count = 0
        for score in result_list:
            feature_file_list.append(byte_to_string(score[self._features_filename_key]))
            feature_list.append(score[self._features_embedding_key].reshape(1, self.params.dimension))
            count += 1
            if count % 100 == 0:
                self.logger.info("calculated {} image...".format(count))

        if count % 100 != 0:
            self.logger.info("calculated {} image...".format(count))

        self.logger.info("success to list feature for {} file!".format(len(input_list)))

        return feature_list, feature_file_list

    def show_embedding(self, mode: ProcessMode, count: int = -1, remove_log_dir_if_exists: bool = False):
        key_name = "train" if mode == ProcessMode.train else "test"
        log_dir = os.path.join(self.TRAIN_CKPT_DIR, "logs_{}".format(key_name))
        if os.path.exists(log_dir) and remove_log_dir_if_exists:
            shutil.rmtree(log_dir)

        self.logger.info("trying to show {} embedding...".format(key_name))

        image_list, label_list = self.get_file_list(
            split_name=DatasetUtils.SPLIT_TRAIN if mode == ProcessMode.train else DatasetUtils.SPLIT_TEST)
        if count > 1:
            image_list = image_list[:count]
            label_list = label_list[:count]

        feature_list, _ = self.list_features(file_list=[image_list, label_list])

        self.logger.info("got {}-image feature: {}/{}".format(
            key_name, len(feature_list), len(image_list))
        )

        show_embedding(feature_list=feature_list, labels=label_list, log_dir=log_dir)
        self.logger.info("success to show {} embedding!".format(key_name))

    def get_dataset_func(self, split_name, num_epochs=1, shuffle=True, batch_size=64, num_parallel_calls=2,
                         prefetch_size=2, shuffle_size=4, input_list=None):
        _shuffle = False
        _num_epochs = 1

        if input_list:
            train_data, train_labels = input_list
        elif split_name == DatasetUtils.SPLIT_TRAIN:
            train_data, train_labels = self.get_file_list(split_name=DatasetUtils.SPLIT_TRAIN)
            _shuffle, _num_epochs = True, num_epochs
        else:
            train_data, train_labels = self.get_file_list(split_name=DatasetUtils.SPLIT_TEST)

        file_name_list = np.array([""] * len(train_labels), dtype=str).reshape((len(train_labels),))
        return tf.estimator.inputs.numpy_input_fn(
            x={self._features_images_key: train_data, self._features_filename_key: file_name_list},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=_num_epochs,
            shuffle=_shuffle,
            queue_capacity=min(batch_size * 4, 2048 * 16),
            num_threads=1
        )


class ArcFaceLossModelMnist(TripletLossModelMnist):
    def __init__(self, train_ckpt_dir, pretrained_ckpt_file: str = None):
        super(ArcFaceLossModelMnist, self).__init__(
            train_ckpt_dir=train_ckpt_dir,
            pretrained_ckpt_file=pretrained_ckpt_file
        )
        self.model_name = "ArcFaceLoss"

        self.params = Params({
            "num_channels": 1,
            "num_filter": 32,
            "num_class": 10,
            "margin": 1.0,
            "dimension": 64,
            "image_size": 28,
        })

    def model_fun(self, ):
        """ 返回func """

        def get_model_net(scope_name, features, params, is_training, labels=None):
            inputs = tf.reshape(features[self._features_images_key],
                                [-1, params.image_size, params.image_size, params.num_channels])

            with tf.variable_scope(scope_name, 'arc_face_loss', [inputs]) as sc:
                out = inputs
                num_filter = params.num_filter
                channels = [num_filter, num_filter * 2]
                for i, c in enumerate(channels):
                    with tf.variable_scope('block_{}'.format(i + 1)):
                        out = tf.layers.conv2d(out, c, 3, padding='same')
                        out = tf.layers.batch_normalization(out, momentum=0.9, training=is_training)
                        out = tf.nn.relu(out)
                        out = tf.layers.max_pooling2d(out, 2, 2)

                assert out.shape[1:] == [7, 7, num_filter * 2]

                out = tf.reshape(out, [-1, 7 * 7 * num_filter * 2])
                with tf.variable_scope('fc_1'):
                    embeddings = tf.layers.dense(out, params.dimension)

                end_points = {End_Point_Prediction_Key: {self._features_embedding_key: embeddings}}

                if labels is None:
                    logits = embeddings
                else:
                    logits = face_losses.arcface_loss_layer(
                        embeddings=embeddings, labels=labels, out_num=params.num_class, )

                return logits, end_points

        return tf_model_fn.tf_softmax_model_fn(
            network=get_model_net,
            scope_name=self.model_name,
            features_filename_key=self._features_filename_key,
            get_learning_rate_func=self.get_learning_rate,
            optimizer_type=OptimizerType.adam,
            logger=self.logger
        )


__all__ = ("TripletLossModelMnist", "ArcFaceLossModelMnist")

if __name__ == '__main__':
    init_logger()
    estimator = ArcFaceLossModelMnist("./arcfacex3", tf.train.latest_checkpoint("./arcfacex1"))
    for _ in range(50):
        estimator.train(batch_size=512, num_epochs=2)
    estimator.show_embedding(mode=ProcessMode.test, count=2000,
                             remove_log_dir_if_exists=True)
