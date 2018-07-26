##################################################################
#        MESURE DU TAUX D'OCCUPATION DE PARKINGS A L'AIDE        #
#                       DE CAMERAS VIDEOS                        #
# -------------------------------------------------------------- #
#               Rémi Jacquemard - TB 2018 - HEIG-VD              #
#                   remi.jacquemard@heig-vd.ch                   #
#                           July 2018                            #
# -------------------------------------------------------------- #
# Used to label the cars dataset for the tensorflow object       #
# detection API.                                                 #
##################################################################

import scipy.io as sio
from pathlib import Path
from skimage import io
from xml.etree import ElementTree 
import sys, os

HOME_PATH = str(Path.home())

MAT_FILE = HOME_PATH + "/DS/cars/cars_annos.mat"
CARS_PATH = HOME_PATH + "/DS/cars/images"
ANNOTATION_PATH = HOME_PATH + "/DS/cars/annotations"
os.makedirs(ANNOTATION_PATH, exist_ok=True)
XML_PATH = HOME_PATH + "/DS/cars/annotations/xmls"
os.makedirs(XML_PATH, exist_ok=True)


# loading the mat (MATLAB) file
content = sio.loadmat(MAT_FILE)

# creating the trainval text file
trainval_file = open(ANNOTATION_PATH + "/trainval.txt", "w")

for annotation in content['annotations'][0]:
    # getting info
    filename = annotation[0][0].split('/')[-1]

    xmin = str(annotation[1][0][0])
    ymin = str(annotation[2][0][0])
    xmax = str(annotation[3][0][0])
    ymax = str(annotation[4][0][0])

    image = io.imread(CARS_PATH + "/" + filename)

    # taking only colored image
    if len(image.shape) < 3:
        print(filename + " is not a tri-channel image")
        continue
    
    #print("{}:{},{},{},{}".format(filename, xmin, ymin, xmax, ymax))

    # creating xml VOC file
    root = ElementTree.Element("annotation")
    # -- filename
    filename_elem = ElementTree.Element("filename")
    filename_elem.text = filename
    root.append(filename_elem)
    # -- size
    size_elem = ElementTree.Element("size")
    root.append(size_elem)
    width_elem = ElementTree.Element("width")
    width_elem.text = str(image.shape[1])
    size_elem.append(width_elem)
    height_elem = ElementTree.Element("height")
    height_elem.text = str(image.shape[0])
    size_elem.append(height_elem)
    depth_elem = ElementTree.Element("depth")
    depth_elem.text = str(image.shape[2])
    size_elem.append(depth_elem)
    # -- segmented
    segmented_elem = ElementTree.Element("segmented")
    segmented_elem.text = str(0)
    root.append(segmented_elem)
    # -- car object
    object_element = ElementTree.Element("object")
    root.append(object_element)

    name_elem = ElementTree.Element("name")
    name_elem.text = "car"
    object_element.append(name_elem)

    bounds_elem = ElementTree.Element("bndbox")
    object_element.append(bounds_elem)

    xmin_elem = ElementTree.Element("xmin")
    xmin_elem.text = xmin
    bounds_elem.append(xmin_elem)
    ymin_elem = ElementTree.Element("ymin")
    ymin_elem.text = ymin
    bounds_elem.append(ymin_elem)
    xmax_elem = ElementTree.Element("xmax")
    xmax_elem.text = xmax
    bounds_elem.append(xmax_elem)
    ymax_elem = ElementTree.Element("ymax")
    ymax_elem.text = ymax
    bounds_elem.append(ymax_elem)

    tree = ElementTree.ElementTree(root)
    tree.write(XML_PATH + "/" + filename.split('.')[0] + ".xml")

    # Adding the image to trainvals
    trainval_file.write(filename.split('.')[0] + "\n")

trainval_file.close()