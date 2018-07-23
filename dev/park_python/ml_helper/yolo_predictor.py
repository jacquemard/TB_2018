import darknet

class YoloPredictor:
    CONFIG_PATH = "yolov3.cfg"
    META_PATH = "yolov3.data"

    def __init__(self, weights):
        self.net = darknet.load_net(self.CONFIG_PATH, weights, 0)
        self.meta = darknet.load_meta(self.META_PATH)

    def predict_image_file(self, image_file):
        detection = darknet.detect(self.net, self.meta, image_file)
        print(detection)
