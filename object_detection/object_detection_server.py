# -*- coding:utf-8 -*-
from __future__ import absolute_import

import numpy as np
import os
import tarfile
import tensorflow as tf

from ml_tools.tf_utils import get_wsl_path
from ml_tools.object_detection import label_map_util
from ml_tools.object_detection.utils import visualization_utils as vis_util


class ObjectDetetion(object):
    # What model to download.
    ROOT_PATH = get_wsl_path("E:/frkhit/Download/AI/pre-trained-model/object_detection")
    MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
    MODEL_FILE = os.path.join(ROOT_PATH, "{}.tar.gz".format(MODEL_NAME))
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = os.path.join(ROOT_PATH, MODEL_NAME, "frozen_inference_graph.pb")

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(os.path.dirname(__file__), "data/mscoco_label_map.pbtxt")

    NUM_CLASSES = 90

    def __init__(self):
        tar_file = tarfile.open(self.MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, self.ROOT_PATH)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.detection_graph = detection_graph
        self.sess = None

    @staticmethod
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def api(self, image):
        if self.sess is None:
            self.sess = tf.Session(graph=self.detection_graph)
            detection_graph = self.detection_graph
            # Definite input and output Tensors for detection_graph
            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # todo save graph
            # self._save_graph(self.sess, "logs/faster_02")

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        return image_np

    @staticmethod
    def _save_graph(session, path):
        # path exists
        if not os.path.exists(path):
            os.mkdir(path)

        # graph
        train_writer = tf.summary.FileWriter(path, session.graph)
        train_writer.close()


if __name__ == '__main__':
    from PIL import Image

    image_output = ObjectDetetion().api(Image.open("./a.jpg"))
    im = Image.fromarray(image_output)
    im.save(get_wsl_path("E:/ac.jpg"))
