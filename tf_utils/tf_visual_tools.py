# -*- coding:utf-8 -*-
from __future__ import absolute_import

import os
import shutil
import tensorflow as tf

from ml_tools.preprocessing.inception_preprocessing import preprocess_image as inception_preprocess_image
from ml_tools.preprocessing.vgg_preprocessing import preprocess_image as resnet_preprocess_image
from ml_tools.preprocessing.vgg_preprocessing import preprocess_image as vgg_preprocess_image
from ml_tools.tf_utils.pre_process_utils import whale_gray_preprocess_image, tf_image_random_gray, \
    whale_rgb_preprocess_image

try:
    from pymltools.tf_utils import *
except ImportError:
    from pymltools.pymltools.tf_utils import *


class ImageBoxIterator(SmartImageIterator):
    def __init__(self, tf_decoder_func=tf_decode_jpg):
        super(ImageBoxIterator, self).__init__(tf_decoder_func=tf_decoder_func)
        self.nan_size = 100000

    def get_tf_image_iterator(self):
        def image_generator():
            for file_name, offset_height, offset_width, target_height, target_width in self._file_list:
                yield (file_name, offset_height, offset_width, target_height, target_width)

        image_generator.out_type = (tf.string, tf.int32, tf.int32, tf.int32, tf.int32)
        image_generator.out_shape = (
            tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))

        if self._iter is None:
            dataset = tf.data.Dataset.from_generator(
                image_generator,
                output_types=image_generator.out_type,
                output_shapes=image_generator.out_shape, )

            dataset = dataset.map(self._tf_decoder_func, num_parallel_calls=2)
            dataset = dataset.batch(self.batch_size)
            self._iter = dataset.make_initializable_iterator()

        return self._iter

    def set_file_list(self, jpg_file_list: list, boxes_list: list = None):
        self._file_list.clear()

        count = len(jpg_file_list)
        if boxes_list:
            assert count == len(boxes_list)
            offset_height_list, offset_width_list, target_height_list, target_width_list = \
                parse_bounding_boxes_list(boxes_list)
        else:
            offset_height_list, offset_width_list, target_height_list, target_width_list = \
                [0] * count, [0] * count, [self.nan_size] * count, [self.nan_size] * count

        for index, jpg_file in enumerate(jpg_file_list):
            self._file_list.append((jpg_file, offset_height_list[index], offset_width_list[index],
                                    target_height_list[index], target_width_list[index]))


class PreProcessFuncVisualTools(object):
    """
        可视化模型前处理过程效果
    """
    tmp_dir = "/tmp/tensorboard"

    def __init__(self, ):
        pass

    @classmethod
    def prepare_dir(cls):
        if os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    @classmethod
    def show_data_process(cls, jpg_file_list: list, preprocess_func, tmp_dir: str = None, max_outputs: int = -1,
                          boxes_list: list = None):
        def _decoder(file_name, offset_height=None, offset_width=None, target_height=None, target_width=None):
            image_str = tf.read_file(file_name)
            image_str = tf.image.decode_jpeg(image_str, channels=3)
            raw_image = tf.image.resize_images(image_str, size=(250, 250))
            if offset_height is not None:
                return raw_image, preprocess_func(image_str, offset_height, offset_width, target_height, target_width)

            return raw_image, preprocess_func(raw_image)  # todo image size change

        if tmp_dir is None:
            tmp_dir = cls.tmp_dir
        if max_outputs <= 0:
            max_outputs = len(jpg_file_list)

        smart_iterator = ImageBoxIterator(tf_decoder_func=_decoder)
        smart_iterator.set_file_list(jpg_file_list=jpg_file_list, boxes_list=boxes_list)

        raw_images, images = smart_iterator.get_tf_image_iterator().get_next()
        tf.summary.image('raw_image', raw_images, max_outputs=max_outputs)
        tf.summary.image('preprocess_image', images, max_outputs=max_outputs)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(smart_iterator.get_tf_image_iterator().initializer)
            summary_writer = tf.summary.FileWriter(tmp_dir, sess.graph)
            while True:
                try:
                    summary_all = sess.run(merged)
                    summary_writer.add_summary(summary_all)
                except tf.errors.OutOfRangeError:
                    break

            summary_writer.close()

        tf.logging.info("show_data_process: tensorboard --logdir={}".format(os.path.abspath(tmp_dir)))

    def show_vgg(self, jpg_file_list: list, is_training: bool = True):
        def process_func(image, ):
            return vgg_preprocess_image(image, 224, 224, is_training=is_training)

        self.show_data_process(jpg_file_list=jpg_file_list, preprocess_func=process_func)

    def show_whale(self, jpg_file_list: list, is_training: bool = True):
        def process_func(image, bbox=None):
            return whale_gray_preprocess_image(image, 384, 384, bbox=bbox, is_training=is_training)

        self.show_data_process(jpg_file_list=jpg_file_list, preprocess_func=process_func)

    def show_whale_rgb(self, jpg_file_list: list, is_training: bool = True, boxes_list: list = None):
        def process_func(image, offset_height=None, offset_width=None, target_height=None, target_width=None):
            if offset_height is not None:
                image = tf_image_crop(image, offset_height, offset_width, target_height, target_width)
                image = tf.image.resize_images(image, size=(1000, 1000))

            return whale_rgb_preprocess_image(image, 384, 384, is_training=is_training)

        self.show_data_process(jpg_file_list=jpg_file_list, preprocess_func=process_func, boxes_list=boxes_list)

    def show_whale_vgg(self, jpg_file_list: list, is_training: bool = True):
        def process_func(image, bbox=None):
            return resnet_preprocess_image(
                tf_image_random_gray(image, gray_prob=0.8),
                224, 224, is_training=is_training)

        self.show_data_process(jpg_file_list=jpg_file_list, preprocess_func=process_func)

    def show_whale_with_crop(self, jpg_file_list: list, bounding_box_list: list, is_training: bool = True):
        def tf_decode_with_crop(file_name, offset_height, offset_width, target_height, target_width):
            # todo not work well
            image_str = tf.read_file(file_name)
            image = tf.image.decode_jpeg(image_str, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            bbox = tf_bounding_box_to_bbox(image, offset_height, offset_width, target_height, target_width)
            image = tf.image.resize_images(image, size=(500, 500))
            processed_images = whale_gray_preprocess_image(image, 384, 384, bbox=bbox, is_training=is_training)
            return image, processed_images, bbox

        def tf_decode_with_crop_v2(file_name, offset_height, offset_width, target_height, target_width):
            image_str = tf.read_file(file_name)
            image = tf.image.decode_jpeg(image_str, channels=3)
            image = tf_image_crop(image, offset_height, offset_width, target_height, target_width)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize_images(image, size=(500, 500))
            processed_images = whale_gray_preprocess_image(image, 384, 384, is_training=is_training)
            return image, processed_images

        tmp_dir = self.tmp_dir
        max_outputs = len(jpg_file_list)

        offset_height_list, offset_width_list, target_height_list, target_width_list = \
            parse_bounding_boxes_list(bounding_box_list)

        dataset = tf.data.Dataset.from_tensor_slices(
            (jpg_file_list, offset_height_list, offset_width_list, target_height_list, target_width_list))
        dataset = dataset.map(tf_decode_with_crop_v2, num_parallel_calls=1)
        dataset = dataset.batch(batch_size=32)

        iterator = dataset.make_initializable_iterator()
        raw_images, images = iterator.get_next()
        raw_images = tf.reshape(raw_images, shape=(-1, 500, 500, 1))
        images = tf.reshape(images, shape=(-1, 384, 384, 1))

        tf.summary.image('raw_image', tf.image.grayscale_to_rgb(raw_images), max_outputs=max_outputs)
        tf.summary.image('preprocess_image', tf.image.grayscale_to_rgb(images), max_outputs=max_outputs)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            summary_writer = tf.summary.FileWriter(tmp_dir, sess.graph)
            while True:
                try:
                    summary_all = sess.run(merged)
                    summary_writer.add_summary(summary_all)
                except tf.errors.OutOfRangeError:
                    break

            summary_writer.close()

        tf.logging.info("show_data_process: tensorboard --logdir={}".format(os.path.abspath(tmp_dir)))

    def show_inception(self, jpg_file_list: list, is_training: bool = True):

        def process_func(image, ):
            return inception_preprocess_image(image, 224, 224, is_training=is_training)

        self.show_data_process(jpg_file_list=jpg_file_list, preprocess_func=process_func)

    @classmethod
    def image_show_demo(cls, jpg_file_list: list):
        smart_iterator = SmartImageIterator(tf_decoder_func=ImageIteratorTestDemo.tf_decode_jpg_with_big_size)
        smart_iterator.set_file_list(jpg_file_list=jpg_file_list)

        images, image_file_names = smart_iterator.get_tf_image_iterator().get_next()
        resized_image = tf.image.resize_images(images, [256, 256], method=tf.image.ResizeMethod.AREA)
        cropped_image = tf.image.crop_to_bounding_box(images, 20, 20, 256, 256)
        flipped_image = tf.image.flip_up_down(images)
        rotated_image = tf.image.rot90(images, k=1)
        grayed_image = tf.image.rgb_to_grayscale(images)
        grg_image = tf.image.grayscale_to_rgb(grayed_image)

        tf.summary.image('image resized', resized_image)
        tf.summary.image('image cropped', cropped_image)
        tf.summary.image('image flipped', flipped_image)
        tf.summary.image('image rotated', rotated_image)
        tf.summary.image('image grayed', grayed_image)
        tf.summary.image('image grg', grg_image)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(smart_iterator.get_tf_image_iterator().initializer)
            summary_writer = tf.summary.FileWriter(cls.tmp_dir, sess.graph)

            while True:
                try:
                    summary_all = sess.run(merged)
                    summary_writer.add_summary(summary_all, 0)
                except tf.errors.OutOfRangeError:
                    break

            summary_writer.close()


__all__ = ("PreProcessFuncVisualTools", "ImageBoxIterator")
