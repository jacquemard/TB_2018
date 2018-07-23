######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Rémi Jacquemard
# Based on Author: Evan Juras
# 

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

from pathlib import Path
import os
import sys
cur_path = Path(os.path.abspath(__file__))
lib_path = str(cur_path.parent.parent.parent.resolve()) + "/tensorflow_models/object_detection"
sys.path.insert(0, lib_path)
lib_path = str(cur_path.parent.parent.parent.resolve()) + "/tensorflow_models"
sys.path.insert(0, lib_path)

# Import packages
import os
from skimage import io
import numpy as np
import tensorflow as tf
import sys
import argparse


# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

class TensorflowPredictor:
    # Path to label map file
    PATH_TO_LABELS = 'tensorflow_label_map.pbtxt'
    
    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Treshold for a region to be a car
    TRESH = 0.7

    def __init__(self, model_file):
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        
        categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        
        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
        
        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def draw_boxes(self, image, boxes, classes, scores):
        # Draw the results of the detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.4)

    def predict_image(self, image):
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        return self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})


    def predict_num_cars(self, image):
        _, scores, _, _ = self.predict_image(image)
        scores = scores[0] 
        num = 0
        for score in scores:
            if score >= self.TRESH:
                num += 1

        return num
       
    
    def predict_image_file(self, image_file, output_image_file):
        image = io.imread(image_file)
        (boxes, scores, classes, _) = self.predict_image(image)
        self.draw_boxes(image, boxes, classes, scores)
        io.imsave(output_image_file,  image)



parser = argparse.ArgumentParser("tensorflow predictor")
parser.add_argument("-m", "--model", nargs=1, help="the tensorflow graph model to use", type=str, required=True)
parser.add_argument("-i", "--input", nargs=1, help="the image file path to predict", type=str, required=True)
parser.add_argument("-o", "--output", nargs=1, help="the image output file", type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    predictor = TensorflowPredictor(args.model[0])

    predictor.predict_image_file(args.input[0], args.output[0])


# Name of the directory containing the object detection module we're using
#MODEL_NAME = '/home/ubuntu/DS/cars/training'
#IMAGE_NAME = '/home/ubuntu/DS/cars/images/000001.jpg'

# Grab path to current working directory
#CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
#PATH_TO_CKPT = '/home/ubuntu/TB/GIT/dev/park_python/drafts/object_detection/tensorflow_api/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'



# Path to image
#PATH_TO_IMAGE = '/home/ubuntu/DS/PKLot/PKLot/UFPR05_annot/24/2013-02-22_17_30_12.bmp'

# Output file
#PATH_TO_OUTPUT = '/home/ubuntu/DS/PKLot/PKLot/UFPR05_annot/24/2013-02-22_17_30_12_output.bmp'



# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine




# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
#image = cv2.imread(PATH_TO_IMAGE)

# All the results have been drawn on image. Now display the image.
#cv2.imshow('Object detector', image)
#cv2.imwrite(PATH_TO_OUTPUT, image)

# Press any key to close the image
#cv2.waitKey(0)

# Clean up
#cv2.destroyAllWindows()
