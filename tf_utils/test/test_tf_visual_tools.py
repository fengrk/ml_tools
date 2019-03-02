# -*- coding:utf-8 -*-
from __future__ import absolute_import

import unittest

import os

from ml_tools.tf_utils import PreProcessFuncVisualTools, EmbeddingVisualTools
from pyxtools import list_files


class TestPreProcessFuncVisualTools(unittest.TestCase):
    def setUp(self):
        self.all_image_list = [
            img_file for img_file in list_files(os.path.dirname(__file__)) if img_file.endswith(".jpg")
        ]

    def tearDown(self):
        pass

    def testDemo(self):
        PreProcessFuncVisualTools.prepare_dir()
        PreProcessFuncVisualTools.image_show_demo(jpg_file_list=self.all_image_list)

    def testVgg(self):
        PreProcessFuncVisualTools.prepare_dir()
        PreProcessFuncVisualTools().show_vgg(jpg_file_list=self.all_image_list, is_training=True)


class TestEmbeddingVisualTools(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testRandom(self):
        EmbeddingVisualTools.show_random_embeddings()
