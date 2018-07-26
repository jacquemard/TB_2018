##################################################################
#        MESURE DU TAUX D'OCCUPATION DE PARKINGS A L'AIDE        #
#                       DE CAMERAS VIDEOS                        #
# -------------------------------------------------------------- #
#              Rémi Jacquemard - TB 2018 - HEIG-VD               #
#                   remi.jacquemard@heig-vd.ch                   #
#               https://github.com/remij1/TB_2018                #
#                           July 2018                            #
# -------------------------------------------------------------- #
# Draft: used to explore different type of image processing.     #
##################################################################

from skimage import io, transform, filters, exposure, util, color
from skimage.viewer.viewers import ImageViewer, CollectionViewer
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import datetime

# Loading the image. This is a numpy array with a [x, y, 3] shape
img = io.imread("park_python/tests/cam_image_processing/park.jpg")

def save_images(images, folder):    
    d = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    folder = folder + "/" + d + "/"
    
    os.makedirs(folder)

    for i, image in enumerate(images):
        io.imsave(folder + str(i) + ".png", image)

    print("Images saved")

# Downsampling tests
# The image is 1280x720px, which is a 16/9 ratio
# Note: img is an 3D array where each line represents an image line. Therefore, accessing
# the first dimension design a line, and accessing the second dimension design a pixel
# in this line. So, for an x*y pixels image, its shape is (y, x, 3).
min_size = (90, 160)

def get_downsampled_images(image):
    current_size = min_size

    downsampled_images = []

    while current_size[0] <= image.shape[0]:
        downsampled_image = transform.resize(image, current_size, mode='reflect')

        downsampled_images.append(downsampled_image)
        
        current_size = (current_size[0] + min_size[0], current_size[1] + min_size[1])

    return downsampled_images

# done for each chanel (adapted from http://scikit-image.org/docs/0.12.x/auto_examples/color_exposure/plot_adapt_rgb.html)    
@adapt_rgb(each_channel)
def fil_each(i, fil):    
    # We use 1 - filters(image)
    # but this will not work if image is not normalized
    return fil(i)

@adapt_rgb(hsv_value)
def fil_hsv(i, fil):
    return fil(i)

def fil_gray(i, fil):
    return fil(color.rgb2gray(i))

def get_edges_images(image):    
    fils = [filters.sobel, filters.scharr]

    funcs = [fil_each, fil_hsv, fil_gray]

    edge_images = []
    for fil in fils:        
        for func in funcs:
            edge_images.append(exposure.rescale_intensity(1 - func(image, fil)))

    return edge_images
"""
imgs = get_edges_images(img)
save_images(imgs, "edge_test_output/edges_only")
#CollectionViewer(imgs).show()
imgs = get_downsampled_images(img)
save_images(imgs, "edge_test_output/downsample_only")
#CollectionViewer(imgs).show()


# ---------- 1) edge, 2) downsample

# The get_edges_images(img)[4] seems better (scharr filter, hsv values)
edge_image = exposure.rescale_intensity(1 - fil_hsv(img, filters.scharr))
# Exploring downsampling
downsampled_images = get_downsampled_images(edge_image)
save_images(downsampled_images, "edge_test_output/edge-downsample")
#CollectionViewer(downsampled_images).show()
# The downsampled_images[2] seems ok (270x480)
"""


# ---------- 1) downsample, 2) edge
# The get_downsampled_images(img)[2] seems ok (270x480)
downsampled_image = transform.resize(img, (270, 480), mode='reflect')
# Exploring edge detections
edge_images = get_edges_images(downsampled_image)
save_images(edge_images, "edge_test_output/downsample-edge")
"""
#CollectionViewer(edge_images).show()
# Scharr hsv filter seems better
# Exploring differents downsampling with scharr hsv filter
images = list(map(lambda i: exposure.rescale_intensity(1 - fil_hsv(i, filters.scharr)), get_downsampled_images(img)))
# displaying images
save_images(downsampled_images, "edge_test_output/downsample-scharredge")

#CollectionViewer(images).show()
# images[2] seems better (270x480), with scharr filter


# ---------- downsample or edge detection first ?

edge_first = exposure.rescale_intensity(1 - fil_hsv(img, filters.scharr))
edge_first = transform.resize(edge_first, (270, 480), mode='reflect')

downsample_first = transform.resize(img, (270, 480), mode='reflect')
downsample_first = exposure.rescale_intensity(1 - fil_hsv(downsample_first, filters.scharr))

save_images([edge_first, downsample_first], "edge_test_output/results_both")

#CollectionViewer([edge_first, downsample_first]).show()
# the downsample_first seems better
"""