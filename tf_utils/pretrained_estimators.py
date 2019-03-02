# -*- coding:utf-8 -*-
from __future__ import absolute_import

import json
import logging
import pickle
from urllib import request

import numpy as np
import os
import tempfile
import tensorflow as tf

from ml_tools.nets import nets_factory
from ml_tools.nets.mobilenet import mobilenet_v2
from ml_tools.preprocessing import preprocessing_factory
from ml_tools.preprocessing.vgg_preprocessing import preprocess_image
from pyxtools import byte_to_string
from pyxtools import list_files, get_md5

try:
    from pymltools.tf_utils import PropVisualTools, get_wsl_path, InitFromPretrainedCheckpointHook, \
        SmartImageIterator, ImageIteratorTestDemo, get_readable_names_for_imagenet_labels
except ImportError:
    from pymltools.pymltools.tf_utils import PropVisualTools, get_wsl_path, InitFromPretrainedCheckpointHook, \
        SmartImageIterator, ImageIteratorTestDemo, get_readable_names_for_imagenet_labels

current_dir = os.path.dirname(__file__)

all_test_image_list = [
    img_file for img_file in list_files(os.path.join(os.path.dirname(__file__), "test")) if
    img_file.endswith(".jpg")
]


class FeatureExtractorEstimator(object):
    default_tmp_dir = os.path.join(tempfile.gettempdir(), "FeatureExtractorEstimator")
    class_id_vs_name = {}

    def __init__(self, network_name, checkpoint_path, batch_size, num_classes, layer_names: list,
                 image_size=None, preproc_func_name=None, preproc_threads=2, image_preproc_fn=None,
                 is_training: bool = False):
        """
        TensorFlow feature extractor using tf.slim and models/slim.
        Core functionalities are loading network architecture, pretrained weights,
        setting up an image pre-processing function, queues for fast input reading.
        The main workflow after initialization is first loading a list of image
        files using the `enqueue_image_files` function and then pushing them
        through the network with `feed_forward_batch`.

        For pre-trained networks and some more explanation, checkout:
          https://github.com/tensorflow/models/tree/master/slim

        :param network_name: str, network name (e.g. resnet_v1_101)
        :param checkpoint_path: str, full path to checkpoint file to load
        :param batch_size: int, batch size
        :param layer_names: list, layer name in end points
        :param num_classes: int, number of output classes
        :param image_size: int, width and height to overrule default_image_size (default=224)
        :param preproc_func_name: str, optional to overwrite default processing (default=None)
        :param preproc_threads: int, number of input threads (default=1)
        :param image_preproc_fn: func, optional to overwrite default processing (default=None)
        :param is_training: bool, is training

        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.preproc_threads = preproc_threads
        self.image_preproc_fn = image_preproc_fn
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.network_name = network_name
        self.layer_names = layer_names
        self.is_training = is_training

        # estimator setting
        self._key_image = "image"
        self._key_file_name = "file"

        # model_fun
        self.network_fn = nets_factory.get_network_fn(self.network_name, num_classes=self.num_classes,
                                                      is_training=is_training)
        self.image_size = image_size or self.network_fn.default_image_size

        # image_preproc_fn
        if self.image_preproc_fn is None:
            preproc_func_name = self.network_name if preproc_func_name is None else preproc_func_name
            self.image_preproc_fn = preprocessing_factory.get_preprocessing(preproc_func_name, is_training=is_training)

        # end points
        self._end_points_info = ""

    def get_model_fn(self):

        def model_func(features, labels, mode):
            input_image = tf.reshape(features[self._key_image], [-1, self.image_size, self.image_size, 3])

            net, end_points = self.network_fn(input_image, )
            if not self._end_points_info:
                info_str_list = ["", "num_classes is {}".format(self.num_classes)]
                for key, value in end_points.items():
                    info_str_list.append("{}[{}]->{}".format(key, value.shape, value))

                self._end_points_info = "\n".join(info_str_list)

            predictions = {self._key_file_name: features[self._key_file_name], }
            for layer in self.layer_names:
                predictions[layer] = end_points[layer]
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        return model_func

    def get_dataset_func(self, jpg_file_list: list, ):
        def decoder(file_name, label):
            image = tf.read_file(file_name)
            image = tf.image.decode_jpeg(image, channels=3)
            image = self.image_preproc_fn(image, self.image_size, self.image_size)
            return {self._key_image: image, self._key_file_name: file_name}, label

        def input_fn():
            labels = tf.constant([0] * len(jpg_file_list))
            dataset = tf.data.Dataset.from_tensor_slices((jpg_file_list, labels))
            dataset = dataset.map(decoder, num_parallel_calls=self.preproc_threads)
            dataset = dataset.prefetch(buffer_size=2 * self.batch_size)
            dataset = dataset.batch(self.batch_size)

            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels

        return input_fn

    def list_feature(self, jpg_file_list: list, ) -> dict:
        """
        :param jpg_file_list:
        :return:
        """
        train_hooks = [InitFromPretrainedCheckpointHook(self.checkpoint_path, exclusion_list=["global_step"])]

        classifier = tf.estimator.Estimator(model_fn=self.get_model_fn(), model_dir=self.default_tmp_dir)

        scores = classifier.predict(input_fn=self.get_dataset_func(jpg_file_list=jpg_file_list),
                                    hooks=None if not train_hooks else train_hooks, )

        data = {layer_name: [] for layer_name in self.layer_names}
        data[self._key_file_name] = []
        for score in scores:
            data[self._key_file_name].append(byte_to_string(score[self._key_file_name]))
            for layer_name in self.layer_names:
                data[layer_name].append(score[layer_name])

        return data

    @property
    def key_file_name(self) -> str:
        return self._key_file_name

    def print_network_summary(self):
        if self._end_points_info:
            self.logger.info(self._end_points_info)
        else:
            self.logger.warning("<print_network_summary> must run after <list_feature>!")

    @classmethod
    def prepare_label_file(cls):
        uid = os.path.join(tempfile.gettempdir(), get_md5(cls.__class__.__name__.encode("utf-8")) + ".pkl")
        url = "https://raw.githubusercontent.com/frkhit/file_servers/master/imagenet_class_index.json"
        if not os.path.exists(uid):
            with open(uid, "wb") as fw:
                pickle.dump(json.loads(request.urlopen(url).read()), fw)

        with open(uid, "rb") as fr:
            imagenet_info = pickle.load(fr)

        cls.class_id_vs_name = {int(class_id): class_info[1] for class_id, class_info in imagenet_info.items()}

    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        if not cls.class_id_vs_name:
            cls.prepare_label_file()

        return cls.class_id_vs_name[class_id]

    @classmethod
    def num_of_class(cls) -> int:
        if not cls.class_id_vs_name:
            cls.prepare_label_file()

        return len(cls.class_id_vs_name) + 1


class NetInfoParser(object):
    model = None
    num_classes = 1001

    def __init__(self, pre_trained_ckpt: str, ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pre_trained_ckpt = pre_trained_ckpt
        self.smart_iterator = None

    def get_endpoints(self) -> dict:
        raise NotImplementedError

    def print_endpoints(self):
        end_points = self.get_endpoints()
        info_str_list = ["", "num_classes is {}".format(self.num_classes)]
        for key, value in end_points.items():
            info_str_list.append("{}[{}]->{}".format(key, value.shape, value))

        self.logger.info("\n".join(info_str_list))

    def get_image_iterator(self):
        if self.smart_iterator is None:
            all_image_list = [
                img_file for img_file in list_files(os.path.join(os.path.dirname(__file__), "test")) if
                img_file.endswith(".jpg")
            ]

            smart_iter = SmartImageIterator(tf_decoder_func=ImageIteratorTestDemo.tf_decode_jpg_with_size)
            smart_iter.set_file_list(all_image_list)
            self.smart_iterator = smart_iter

        return self.smart_iterator.get_tf_image_iterator()


class NetInfoMobileNetV2(NetInfoParser):
    model = "MobilenetV2"

    def __init__(self, pre_trained_ckpt: str = None):
        if pre_trained_ckpt is None:
            pre_trained_ckpt = get_wsl_path(
                "E:/frkhit/Download/AI/pre-trained-model/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt"
            )
        super(NetInfoMobileNetV2, self).__init__(pre_trained_ckpt)

    def get_endpoints(self):
        image_iterator = self.get_image_iterator()
        images, image_file_names = image_iterator.get_next()

        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
            _, endpoints = mobilenet_v2.mobilenet(images, num_classes=self.num_classes)

        return endpoints


class ImageNetPredict(object):
    model_name = None
    num_classes = 1001

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict_images(self, file_list: list) -> list:
        self.logger.info("trying to predict {} image, first10 is {}".format(len(file_list), file_list[:10]))
        return self._predict_images(file_list)

    def list_features(self, file_list: list, feature_layer: str) -> list:
        self.logger.info("trying to list feature of {} image, first10 is {}".format(len(file_list), file_list[:10]))
        return self._list_features(file_list, feature_layer)

    def list_features_by_queue(self, image_iterator, feature_layer: str) -> (list, list):
        self.logger.info("trying to list feature by queue...")
        return self._list_features_by_queue(image_iterator, feature_layer)

    def _predict_images(self, file_list: list) -> list:
        raise NotImplemented

    def _list_features(self, file_list: list, feature_layer: str) -> list:
        raise NotImplemented

    def _list_features_by_queue(self, image_iterator, feature_layer: str) -> (list, list):
        raise NotImplementedError


class MobileNetV2Predict(ImageNetPredict):
    model_name = NetInfoMobileNetV2.model
    num_classes = NetInfoMobileNetV2.num_classes

    def __init__(self):
        super(MobileNetV2Predict, self).__init__()
        self.pretrained_ckpt_file = get_wsl_path(
            "E:/frkhit/Download/AI/pre-trained-model/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt"
        )
        self.image_size = 224
        self.logger.info("num of class is {}".format(FeatureExtractorEstimator.num_of_class()))
        self._queue_cache = None

    def _load_net(self):
        tf.reset_default_graph()

        # For simplicity we just decode jpeg inside tensorflow.
        # But one can provide any input obviously.
        file_input = tf.placeholder(tf.string, ())

        image = tf.image.decode_jpeg(tf.read_file(file_input))

        images = tf.expand_dims(image, 0)
        images = tf.cast(images, tf.float32) / 128. - 1
        images.set_shape((None, None, None, 3))
        images = tf.image.resize_images(images, (self.image_size, self.image_size))

        # Note: arg_scope is optional for inference.
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
            logits, endpoints = mobilenet_v2.mobilenet(images, num_classes=self.num_classes)

        # Restore using exponential moving average since it produces (1.5-2%) higher
        # accuracy
        ema = tf.train.ExponentialMovingAverage(0.999)
        saver = tf.train.Saver(ema.variables_to_restore())

        return logits, endpoints, saver, file_input, images

    def _predict_images(self, file_list: list) -> list:
        _, endpoints, saver, file_input, _ = self._load_net()

        result_list = []
        with tf.Session() as sess:
            saver.restore(sess, self.pretrained_ckpt_file)
            label_map = get_readable_names_for_imagenet_labels()
            for image_file_name in file_list:
                x = endpoints['Predictions'].eval(feed_dict={file_input: image_file_name})
                result_list.append({
                    "class": int(x.argmax()),
                    "prob": float(x.max()),
                    "class_name": label_map[x.argmax()]
                })

        return result_list

    def prob_visual_hot_graph(self, image_file: str, save_image_file: str):
        _, endpoints, saver, file_input, _ = self._load_net()

        with tf.Session() as sess:
            saver.restore(sess, self.pretrained_ckpt_file)
            tools = PropVisualTools(
                image_file_path=image_file,
                sess=sess,
                file_input=file_input,
                prob_logit=endpoints['Predictions'],
                inner_layer=endpoints["layer_18/output"],
                label_map=get_readable_names_for_imagenet_labels()
            )
            tools.run_expr()
            tools.create_hot_graph(save_image_file)

    def _list_features(self, file_list: list, feature_layer: str) -> list:
        _, endpoints, saver, file_input, images_var = self._load_net()
        if feature_layer not in endpoints:
            self.logger.info("key of endpoints is {}".format(endpoints.keys()))
            raise ValueError("feature layer not in endpoints!")

        result_list = []
        with tf.Session() as sess:
            saver.restore(sess, self.pretrained_ckpt_file)
            for image_file_name in file_list:
                try:
                    x = endpoints[feature_layer].eval(feed_dict={file_input: image_file_name})
                    result_list.append(x)
                except Exception as e:
                    self.logger.error("fail to list feature of {}, error is {}".format(image_file_name, str(e)))
                    # os.remove(image_file_name)
                    result_list.append(np.zeros(result_list[-1].shape, dtype=np.float32))

        return result_list

    def _list_features_by_queue(self, image_iterator, feature_layer: str = "layer_18/output") -> (list, list):
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


class ResNetV250Extractor(object):
    network_name = model = "resnet_v2_50"
    num_classes = FeatureExtractorEstimator.num_of_class()

    def __init__(self, ckpt_file: str = None, batch_size: int = 8,
                 get_preprocess_fn=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ckpt_file = ckpt_file or get_wsl_path(
            "E:/frkhit/Download/AI/pre-trained-model/resnet_v2_50.ckpt")
        self.batch_size = batch_size
        self.layer_names = ["global_pool"]
        self.get_preprocess_fn = get_preprocess_fn

    def list_feature(self, jpg_file_list: list, is_training: bool = False, layers: list = None) -> (dict, list):
        """
            "resnet_v2_50/block4": 7x7x2048
            "global_pool": 1x1x2048

        Args:
            jpg_file_list (list):
            is_training (bool):
            layers (list):
        """
        layers = layers or self.layer_names
        feature_extractor = FeatureExtractorEstimator(
            network_name=self.network_name,
            checkpoint_path=self.ckpt_file,
            batch_size=self.batch_size,
            num_classes=self.num_classes,
            preproc_threads=2,
            layer_names=layers,
            is_training=is_training,
            image_preproc_fn=self.get_preprocess_fn(is_training) if self.get_preprocess_fn else None,
        )

        feature_dataset = feature_extractor.list_feature(jpg_file_list=jpg_file_list, )
        logging.info("Success to exact feature of {} images".format(len(jpg_file_list)))

        feature_data = {layer: [] for layer in layers}
        feature_file_list = []
        for index, image_file in enumerate(feature_dataset[feature_extractor.key_file_name]):
            feature_file_list.append(image_file)
            for layer in layers:
                feature_data[layer].append(feature_dataset[layer][index])

        return feature_data, feature_file_list

    @staticmethod
    def get_preprocess_func(is_training: bool):
        def preprocessing_fn(image, output_height, output_width, **kwargs):
            return preprocess_image(
                image, output_height, output_width,
                is_training=is_training, **kwargs)

        return preprocessing_fn

    @staticmethod
    def get_preprocess_func_for_whale(is_training: bool):
        def preprocessing_fn(image, output_height, output_width, **kwargs):
            image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
            return preprocess_image(
                image, output_height, output_width,
                is_training=is_training, **kwargs)

        return preprocessing_fn

    def print_endpoints(self):
        feature_extractor = FeatureExtractorEstimator(
            network_name=self.network_name,
            checkpoint_path=self.ckpt_file,
            batch_size=self.batch_size,
            num_classes=self.num_classes,
            preproc_threads=1,
            layer_names=self.layer_names,
            is_training=False,
            image_preproc_fn=self.get_preprocess_fn(False) if self.get_preprocess_fn else None,
        )
        feature_extractor.list_feature(jpg_file_list=all_test_image_list[:1])
        feature_extractor.print_network_summary()

    def predict_images(self, jpg_file_list: list) -> list:
        self.logger.info("trying to predict {} image, first10 is {}".format(len(jpg_file_list), jpg_file_list[:10]))

        layer_name = "predictions"
        feature_dict, feature_file_list = self.list_feature(
            jpg_file_list=jpg_file_list, is_training=False, layers=[layer_name])

        result_list = []
        label_map = get_readable_names_for_imagenet_labels()
        for index, image_file_name in enumerate(feature_file_list):
            result_list.append({
                "class": int(feature_dict[layer_name][index].argmax()),
                "prob": float(feature_dict[layer_name][index].max()),
                "class_name": label_map[feature_dict[layer_name][index].argmax()],
                "file_name": image_file_name,
            })

        return result_list


__all__ = ("FeatureExtractorEstimator", "ImageNetPredict", "MobileNetV2Predict",
           "NetInfoParser", "NetInfoMobileNetV2", "ResNetV250Extractor")
