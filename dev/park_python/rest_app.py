from pathlib import Path
import os
import sys
cur_path = Path(os.path.abspath(__file__))

from camera.capture import CameraClient, CameraAgent
from ml_helper.tensorflow_api_predictor import TensorflowPredictor
from flask import Flask, jsonify
from skimage import io
from time import gmtime, strftime

CAMERA_HOST = "ipcam.einet.ad.eivd.ch"
USERNAME = "admin"
PASSWORD = "Lfg3hgPhLdNYW"

MODEL_FILE = str(cur_path.parent.resolve()) + "/final_models/tensorflow/pklotfull_4000_frozen_graph.pb"

MAX_NUM_CARS = 40

VALUES_LENGTH = 4

# Creating a camera client
camera = CameraClient(CAMERA_HOST, USERNAME, PASSWORD)

# Creating the tensorflow predictor
predictor = TensorflowPredictor(MODEL_FILE)

# Agent wich will be called to save park values
values = []

def add_value(value):
    values.insert(0, value)
    if len(values) > VALUES_LENGTH:
        values.pop()

def current_num():
    if len(values) == 0:
        return 0

    return round(sum(values) / len(values))

def image_received(image):
    image = io.imread(image)
    num_cars = predictor.predict_num_cars(image)

    print("image received, num_car: {}, mean: {}".format(num_cars, current_num()))
    add_value(num_cars)

agent = CameraAgent(camera, image_received, seconds=15, blocking=False)

app = Flask(__name__)

@app.route("/")
def root():
    obj = {
        'date':strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        'num_cars': current_num()
    }
    return jsonify(obj)


if __name__ == '__main__':
    agent.start()
    app.run(debug=True)