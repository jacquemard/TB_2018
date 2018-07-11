import numpy as np
import math
from lxml import etree
import os

from pathlib import Path
import sys
cur_path = Path(__file__)
lib_path = str(cur_path.parent.parent.resolve())
sys.path.insert(0, lib_path)
import camera.image_processing as process

from path import Path


def get_cars_grid(image_size, grid_size, xml_file):
    # parsing the xml
    tree = etree.parse(xml_file)

    # finding cars
    cars = tree.xpath("/parking/space[@occupied='1']/rotatedRect/center")
    
    # 2D array representing the grid
    grid = np.zeros((grid_size, grid_size))

    # dimension of a cell
    cell_dim = (math.ceil(image_size[0] / grid_size), math.ceil(image_size[1] / grid_size))

    # mapping the cars to the grid
    for car in cars:
        pixel_x, pixel_y = (int(car.attrib['x']), int(car.attrib['y']))
        grid_x, grid_y = (math.floor(pixel_x / cell_dim[1]), math.floor(pixel_y / cell_dim[0]))

        grid[grid_y][grid_x] = 1.0

    return grid

'''
def get_cars_grid_no_padding(image_size, grid_size, xml_file):
    # parsing the xml
    tree = etree.parse(xml_file)

    # finding cars
    cars = tree.xpath("/parking/space[@occupied='1']/rotatedRect/center")
    
    # 2D array representing the grid
    grid = np.zeros((grid_size, grid_size))

    # dimension of a cell
    cell_dim = (math.floor(image_size[0] / grid_size), math.floor(image_size[1] / grid_size))

    # mapping the cars to the grid
    for car in cars:
        pixel_x, pixel_y = (int(car.attrib['x']), int(car.attrib['y']))
        grid_x, grid_y = (math.floor(pixel_x / cell_dim[1]), math.floor(pixel_y / cell_dim[0]))

        grid[grid_y][grid_x] = 1.0

    return grid
'''


def count_cars(xml_file):
    # parsing the xml
    tree = etree.parse(xml_file)

    # counting cars
    cars = tree.xpath("/parking/space[@occupied='1']")
    return len(cars)

def images_with_xml(dataset_path):
    IMAGE_EXT = ".jpg"
    images = [ 
        (os.path.join(root, f), os.path.join(root, f[:-len(IMAGE_EXT)] + ".xml"))
        for root, _, files in os.walk(dataset_path) 
        for f in files 
        if f.endswith(IMAGE_EXT)]

    return images

def processed_images_with_xml(dataset_path):
    IMAGE_EXT = "_processed.bmp"
    images = [ 
        (os.path.join(root, f), os.path.join(root, f[:-len(IMAGE_EXT)] + ".xml"))
        for root, _, files in os.walk(dataset_path) 
        for f in files 
        if f.endswith(IMAGE_EXT)]

    return images