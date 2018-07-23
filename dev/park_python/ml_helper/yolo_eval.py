from yolo_predictor import YoloPredictor
import os
from tensorflow_api_predictor import TensorflowPredictor
from dataset_helper import pklot

IMAGES_PATH = "/home/ubuntu/DS/PKLot/tensorflow_ds/images_splitted/test"
IMAGE_EXT = ".jpg"
MODEL_PATH = "../final_models/yolo/yolov3_orig.weights"
RESULT_PATH = "yolo_eval.csv"
XMLS_PATH = "/home/ubuntu/DS/PKLot/tensorflow_ds/annotations/xmls"

predictor = YoloPredictor(MODEL_PATH)
result = open(RESULT_PATH, "w")

for root, _, files in os.walk(IMAGES_PATH):
    for f in files:
        if f.endswith(IMAGE_EXT):
            path = os.path.join(root, f)

            xml_filename = f[:-len(IMAGE_EXT)] + ".xml"
            xml = os.path.join(XMLS_PATH, xml_filename)

            n_cars = predictor.predict_num_cars(path)
            real_n_cars = pklot.count_cars(xml)

            result.write("{},{}\n".format(n_cars, real_n_cars))
            print("{} - predicted: {}, wanted:{}".format(f, n_cars, real_n_cars))

result.close()
