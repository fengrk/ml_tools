# -*- coding:utf-8 -*-
from __future__ import absolute_import

import unittest

from ml_tools.tf_utils import ResNetV250Extractor, init_logger
from ml_tools.tf_utils.test.utils import list_jpg_file

init_logger(None)


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testResNetV250Estimator(self):
        extractor = ResNetV250Extractor()
        image_list = list_jpg_file()
        feature_dict, file_list = extractor.list_feature(jpg_file_list=image_list, is_training=False)
        feature_list = feature_dict[extractor.layer_names[0]]
        self.assertEqual(len(file_list), len(image_list))

        feature_dict, file_list_2 = extractor.list_feature(jpg_file_list=image_list, is_training=False)
        feature_list_2 = feature_dict[extractor.layer_names[0]]
        self.assertEqual(len(file_list_2), len(image_list))
        for index in range(len(image_list)):
            self.assertTrue((feature_list[index] == feature_list_2[index]).all())

        feature_dict, file_list_3 = extractor.list_feature(jpg_file_list=image_list, is_training=True)
        feature_list_3 = feature_dict[extractor.layer_names[0]]
        self.assertEqual(len(file_list_3), len(image_list))
        for index in range(len(image_list)):
            self.assertFalse((feature_list_2[index] == feature_list_3[index]).all())

        feature_dict, file_list_4 = extractor.list_feature(jpg_file_list=image_list, is_training=True)
        feature_list_4 = feature_dict[extractor.layer_names[0]]
        self.assertEqual(len(file_list_4), len(image_list))
        for index in range(len(image_list)):
            self.assertFalse((feature_list_2[index] == feature_list_4[index]).all())
            self.assertFalse((feature_list_3[index] == feature_list_4[index]).all())

    def testResNetV250EstimatorPredict(self):
        extractor = ResNetV250Extractor()
        image_list = list_jpg_file()
        result_list = extractor.predict_images(jpg_file_list=image_list)
        self.assertEqual(len(result_list), len(image_list))
        for index in range(len(image_list)):
            self.assertEqual(image_list[index], result_list[index]["file_name"])
