import sys
import os
from pathlib import Path
cur_path = Path(os.path.abspath(__file__) )
lib_path = str(cur_path.parent.parent.resolve())
sys.path.insert(0, lib_path)

from tensorflow_api_predictor import TensorflowPredictor
from skimage import io
from dataset_helper import pklot

IMAGES_PATH = r"C:\Users\Remi\DS\PKLot\PKLot\UFPR05_splitted\test"
IMAGE_EXT = ".jpg"
MODEL_PATH = "../final_models/tensorflow/pklotfull_16000_frozen_graph.pb"
RESULT_PATH = "pklotfull_16000_eval.csv"
XMLS_PATH = r"C:\Users\Remi\DS\PKLot\PKLot\UFPR05_splitted\test"

predictor = TensorflowPredictor(MODEL_PATH)
result = open(RESULT_PATH, "w")

for root, _, files in os.walk(IMAGES_PATH):
    for f in files:
        if f.endswith(IMAGE_EXT):
            path = os.path.join(root, f)
            image = io.imread(path)

            xml_filename = f[:-len(IMAGE_EXT)] + ".xml"
            xml = os.path.join(root, xml_filename)

            n_cars = predictor.predict_num_cars(image)
            real_n_cars = pklot.count_cars(xml)

            result.write("{},{}\n".format(n_cars, real_n_cars))
            print("{} - predicted: {}, wanted:{}".format(f, n_cars, real_n_cars))

result.close()
