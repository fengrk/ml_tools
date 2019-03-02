# -*- coding:utf-8 -*-
from __future__ import absolute_import

import datetime
import logging
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullLocator
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from ml_tools.pytorch_yolov3.models import Darknet
from ml_tools.pytorch_yolov3.utils.datasets import *
from ml_tools.pytorch_yolov3.utils.parse_config import *
from ml_tools.pytorch_yolov3.utils.utils import *
from ml_tools.tf_utils import object_detection_utils
from pyxtools import get_image


def convert_annotation_to_yolo_type(width: int, height: int, xmin: int, ymin: int,
                                    xmax: int, ymax: int) -> (float, float, float, float):
    """
        convert annotation to yolo type

        coco label 格式:
            class_number_1 x y w h
            class_number_2 x y w h

        boxes:
            class_number_1 xmin ymin xmax ymax

    """
    dw = 1. / width
    dh = 1. / height
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_yolo_type_to_annotation(width: int, height: int, x: float, y: float,
                                    w: float, h: float) -> (int, int, int, int):
    """
        convert annotation to yolo type

        coco label 格式:
            class_number_1 x y w h
            class_number_2 x y w h

        boxes:
            class_number_1 xmin ymin xmax ymax

    """
    return int((x - w / 2) * width), int((y - h / 2) * height), int((x + w / 2) * width), int((y + h / 2) * height)


def get_latest_weight(model_dir: str, ):
    all_weight_list = glob.glob(os.path.join(model_dir, "*.weights"))
    if not all_weight_list:
        return None

    w_list = [(int(os.path.basename(_weights_file).split(".")[0]), _weights_file) for _weights_file in
              all_weight_list]

    return sorted(w_list, key=lambda x: x[0], reverse=True)[0][1]


def parse_result(file_name: str, detections: dict, min_score_thresh: float = 0.5,
                 image_size: int = 416) -> list:
    result_list = []
    img = np.array(get_image(file_name))

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (image_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (image_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = image_size - pad_y
    unpad_w = image_size - pad_x

    if detections is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if cls_conf.item() >= min_score_thresh:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                ymin, xmin, ymax, xmax = y1, x1, y1 + box_h, x1 + box_w
                result_list.append(
                    object_detection_utils.detection_data(
                        file_name,
                        int(xmin), int(ymin),
                        int(xmax), int(ymax),
                        float(cls_conf.item())
                    ))

    return result_list


def train_yolov3(class_path: str, data_config_path: str, model_config_path: str, checkpoint_dir: str,
                 use_cuda: bool = True, batch_size: int = 1, epochs: int = 10000, checkpoint_interval: int = 1,
                 n_cpu: int = 0, weights_file: str = None, log_every_steps: int = 100, image_size: int = 416,
                 checkpoint_interval_func=None):
    cuda = torch.cuda.is_available() and use_cuda

    os.makedirs("output", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(data_config_path)
    train_path = data_config["train"]

    classes = load_classes(class_path)

    # Get hyper parameters
    hyperparams = parse_model_config(model_config_path)[0]
    # learning_rate = float(hyperparams["learning_rate"])
    # momentum = float(hyperparams["momentum"])
    # decay = float(hyperparams["decay"])
    # burn_in = int(hyperparams["burn_in"])

    # Initiate model
    model = Darknet(model_config_path, img_size=image_size)

    _weight_file = get_latest_weight(checkpoint_dir)
    if _weight_file:
        model.load_weights(_weight_file)
        logging.info("loading weights from {}".format(_weight_file))
    else:
        if weights_file:
            model.load_weights(weights_file)
            logging.info("loading weights from {}".format(weights_file))
        else:
            model.apply(weights_init_normal)

        logging.info("saving weight...")
        model.save_weights("%s/%d.weights" % (checkpoint_dir, 0))

    if cuda:
        model = model.cuda()

    model.train()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path, img_size=image_size), batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    _steps = 0
    for epoch in range(epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            _steps += 1
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()

            if _steps % log_every_steps == 0:
                logging.info(
                    "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                    % (
                        epoch,
                        epochs,
                        batch_i,
                        len(dataloader),
                        model.losses["x"],
                        model.losses["y"],
                        model.losses["w"],
                        model.losses["h"],
                        model.losses["conf"],
                        model.losses["cls"],
                        loss.item(),
                        model.losses["recall"],
                        model.losses["precision"],
                    )
                )

            model.seen += imgs.size(0)

        if epoch % checkpoint_interval == 0:
            model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))
            if checkpoint_interval_func:
                checkpoint_interval_func()

    if epochs > 1 and (epochs - 1) % checkpoint_interval != 0:
        model.save_weights("%s/%d.weights" % (checkpoint_dir, epochs - 1))


def detect_yolov3(weights_file: str, class_path: str, model_config_path: str, image_folder: str,
                  conf_thres: float = 0.8, use_cuda: bool = True, batch_size: int = 1, nms_thres: float = 0.4,
                  n_cpu: int = 0, output_dir: str = "output", image_size: int = 416, show_image: bool = True, ) -> list:
    cuda = torch.cuda.is_available() and use_cuda

    os.makedirs(output_dir, exist_ok=True)

    # Set up model
    model = Darknet(model_config_path, img_size=image_size)
    model.load_weights(weights_file)

    if cuda:
        model.cuda()

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(ImageFolder(image_folder, img_size=image_size),
                            batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    result_list = []

    logging.info('\nPerforming object detection:')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, len(classes), conf_thres, nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        logging.info('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # result list
    for (path, detections) in zip(imgs, img_detections):
        result_list.extend(
            parse_result(file_name=path, detections=detections, image_size=image_size))

    if show_image:
        # Bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        logging.info('\nSaving images:')
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            logging.info("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # The amount of padding that was added
            pad_x = max(img.shape[0] - img.shape[1], 0) * (image_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (image_size / max(img.shape))
            # Image height and width after padding is removed
            unpad_h = image_size - pad_y
            unpad_w = image_size - pad_x

            # Draw bounding boxes and labels of detections
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    logging.info('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                    # Rescale coordinates to original dimensions
                    box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                    box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                    y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                    x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                             edgecolor=color,
                                             facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                             bbox={'color': color, 'pad': 0})

            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('{}/{}.png'.format(output_dir, img_i), bbox_inches='tight', pad_inches=0.0)
            plt.close()

    # return
    return result_list
