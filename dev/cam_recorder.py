from cam_capture.capture import CameraClient, CameraCrawler
from logger.custom_logger import CustomLogger
from skimage import io
from pathlib import Path
import os
import logging

# ------- CONSTANTS ------- #

CAMERA_HOST = "ipcam.einet.ad.eivd.ch"
USERNAME = "admin"
PASSWORD = "Lfg3hgPhLdNYW"

BASE_FOLDER = "/TB/"

IMAGE_REQUEST_MIN_DELTA = 20
IMAGE_FOLDER = BASE_FOLDER + "output_images/"

MONITORING_MAIL_SUBJECT = "[TB] Monitor - Wanscam Camera Crawler - error or warning occured"
MONITORING_LOG_FOLDER = BASE_FOLDER + "logs/"
MONITORING_LOG_FILE = MONITORING_LOG_FOLDER + "/crawler_log"

# Ensuring that folders exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(MONITORING_LOG_FOLDER, exist_ok=True)

# ------------------------- #

# Creating a camera client
camera = CameraClient(CAMERA_HOST, USERNAME, PASSWORD)

# Defining what to do when an image is received
def handle_image(image):
    # converting to a skimage/opencv image (simply a [x, y, 3] numpy array)
    image = io.imread(image)

# Creating a crawler which request the camera for an image once every 20 minutes
crawler = CameraCrawler(camera, handle_image, minutes = 20)

# Listening for the crawler logs to send email when an error occures
logger = CustomLogger(crawler.get_logger())
logger.set_terminal_handler()
logger.set_file_handler(MONITORING_LOG_FILE)
logger.set_mail_handler(MONITORING_MAIL_SUBJECT)

# Starting the crawler
crawler.start()