# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import argparse
import logging

from ml_tools.pytorch_yolov3.api import train_yolov3
from pyxtools import str2bool, global_init_logger

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=str2bool, default=True, help="whether to use cuda if available")


def train():
    opt = parser.parse_args()
    logging.info(opt)

    train_yolov3(use_cuda=opt.use_cuda, class_path=opt.class_path,
                 model_config_path=opt.model_config_path,
                 batch_size=opt.batch_size, epochs=opt.epochs,
                 data_config_path=opt.data_config_path,
                 checkpoint_interval=opt.checkpoint_interval,
                 checkpoint_dir=opt.checkpoint_dir, n_cpu=opt.n_cpu)


if __name__ == '__main__':
    global_init_logger()
    train()
