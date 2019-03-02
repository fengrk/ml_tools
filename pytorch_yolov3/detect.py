# -*- coding:utf-8 -*-
from __future__ import absolute_import

import argparse
import logging

from ml_tools.pytorch_yolov3.api import detect_yolov3
from pyxtools import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=str2bool, default=True, help='whether to use cuda if available')


def detect():
    opt = parser.parse_args()
    logging.info(opt)
    detect_yolov3(weights_file=opt.weights_path, class_path=opt.class_path, model_config_path=opt.config_path,
                  image_folder=opt.image_folder, conf_thres=opt.conf_thres, use_cuda=opt.use_cuda,
                  batch_size=opt.batch_size, nms_thres=opt.nms_thres, n_cpu=opt.n_cpu, output_dir="output",
                  image_size=opt.img_size)


if __name__ == '__main__':
    import sys

    sys.argv = [
        "x",
        "--use_cuda", "no",
    ]
    detect()
