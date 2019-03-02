# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import time
import unittest

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import normalize

from ml_tools.tf_utils import init_logger, get_triplet_pair_np, load_data_from_h5file
from pyxtools import download_big_file, set_proxy

init_logger(None)


class TestGetTripletPair(unittest.TestCase):
    def setUp(self):
        self.feature = "./feature.h5"
        self.train_csv = "./train.csv"
        if not os.path.exists(self.feature):
            set_proxy("socks5://127.0.0.1:1080")
            download_big_file("https://raw.githubusercontent.com/frkhit/file_servers/master/feature.h5", self.feature)
        assert os.path.exists(self.feature)

        if not os.path.exists(self.train_csv):
            set_proxy("socks5://127.0.0.1:1080")
            download_big_file("https://raw.githubusercontent.com/frkhit/file_servers/master/train.csv", self.train_csv)
        assert os.path.exists(self.train_csv)

    def tearDown(self):
        pass

    def _parse_feature(self):
        # load train info
        whales = pd.read_csv(self.train_csv)
        info = {}
        for index, image_name in enumerate(whales.Image):
            info[image_name] = whales.Id[index]

        # feature
        feature_arr, file_arr = load_data_from_h5file(self.feature, key_list=["feature", "file"])
        feature_arr = normalize(feature_arr, norm='l2', axis=0)  # 对d维分别l2正则化

        file_name_vs_class_id = {}
        class_id_vs_label = {}
        for i in range(file_arr.shape[0]):
            _file_name = file_arr[i].decode("utf-8")
            file_name_vs_class_id[_file_name] = info[os.path.basename(_file_name)]

        # class label
        class_id_list = list(set(file_name_vs_class_id.values()))
        class_id_list.sort()
        for index, class_id in enumerate(class_id_list):
            class_id_vs_label[class_id] = index

        # single_class_id_list
        class_id_vs_file_count = {}
        for class_id in file_name_vs_class_id.values():
            if class_id not in class_id_vs_file_count:
                class_id_vs_file_count[class_id] = 0
            class_id_vs_file_count[class_id] += 1

        single_class_id_list = [class_id for class_id, file_count in class_id_vs_file_count.items() if file_count == 1]

        return feature_arr, file_arr, file_name_vs_class_id, class_id_vs_label, single_class_id_list

    def testAPNTime(self, ):
        feature_arr, file_arr, file_name_vs_class_id, class_id_vs_label, single_class_id_list = self._parse_feature()
        self.assertTrue(len(file_name_vs_class_id) > 0)
        self.assertTrue(len(class_id_vs_label) > 0)

        single_class_id_set = set(single_class_id_list)
        anchor_index_end = -1
        _anchor_search_end = False
        labels = np.zeros(shape=file_arr.shape, dtype=np.int)
        for i in range(file_arr.shape[0]):
            _file_name = file_arr[i].decode("utf-8")
            labels[i] = class_id_vs_label[file_name_vs_class_id[_file_name]]
            if not _anchor_search_end:
                class_id = file_name_vs_class_id[_file_name]
                if class_id in single_class_id_set:
                    _anchor_search_end = True
                else:
                    anchor_index_end = i

        anchor_feature = feature_arr[:anchor_index_end + 1]
        _apn_np_start = time.time()
        apn_list = get_triplet_pair_np(anchor_feature, all_feature=feature_arr, all_label=labels, margin=1.0)
        logging.info("get {} hardest-apn-pairs in {} files, time cost {}s".format(
            len(apn_list), feature_arr.shape[0], time.time() - _apn_np_start))
