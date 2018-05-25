from cam_capture.capture import CameraClient, CameraCrawler
from logger.custom_logger import CustomLogger
import skimage
from skimage import io, transform, exposure, filters
from skimage.color.adapt_rgb import adapt_rgb, hsv_value
from pathlib import Path
import numpy as np
import os
import logging
import timeit
import warnings

# ------- CONSTANTS ------- #

CAMERA_HOST = "ipcam.einet.ad.eivd.ch"
USERNAME = "admin"
PASSWORD = "Lfg3hgPhLdNYW"

BASE_FOLDER = "/TB/"

IMAGE_REQUEST_MIN_DELTA = 20
IMAGE_OUTPUT_SIZE = (270, 480)
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

# 
@adapt_rgb(hsv_value)
def scharr_hsv(image):
    return filters.scharr(image)

# Defining what to do when an image is received
def handle_image(image_stream):
    # Converting to a skimage/opencv image (simply a [x, y, 3] numpy array)
    image = io.imread(image_stream)

    # Firstly, downsampling the image
    image = transform.resize(image, IMAGE_OUTPUT_SIZE, mode='reflect')

    # Secondly, detecting the edges
    image = scharr_hsv(image)

    # Then, we output the image to the folder
    # We use a hash as an ID
    # We could have used a datetime format as a unique filename, but this could results to a lack of privacy
    # Saving it as a bitmap: no decompression to do when loading a lot of file
    # Note: hashing time is negligeable
    h = hash(image.data.tobytes())
    filename = IMAGE_FOLDER + hex(h)[-15:] + ".bmp"

    with warnings.catch_warnings(): # used to ignore loss of precision warning
        warnings.simplefilter("ignore")
        io.imsave(filename, image)

    # We then have to reset metadata access and edit time because of privacy issues
    os.utime(filename, (946684800, 946684800))




# Creating a crawler which request the camera for an image once every 20 minutes
crawler = CameraCrawler(camera, handle_image, minutes = IMAGE_REQUEST_MIN_DELTA)

# Listening for the crawler logs to send email when an error occures
logger = CustomLogger(crawler.get_logger())
logger.set_terminal_handler()
logger.set_file_handler(MONITORING_LOG_FILE)
logger.set_mail_handler(MONITORING_MAIL_SUBJECT)

# Starting the crawler
# This is a blocking instruction
crawler.start()

