import glob
import os
import re
import random
from shutil import copyfile
from pathlib import Path

HOME_PATH = str(Path.home())

DATASET_PATH = HOME_PATH + "/DS/PKLot/PKLot/UFPR05"
OUTPUT_PATH = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_splitted"

TRAIN_PATH = OUTPUT_PATH + "/train"
TEST_PATH = OUTPUT_PATH + "/test"
DEV_PATH = OUTPUT_PATH + "/dev"

TEST_RATE = 0.1
DEV_RATE = 0.1


#shuffling files
files = glob.glob(DATASET_PATH + "/*/*.bmp")
random.shuffle(files)

nb_test = TEST_RATE * len(files)
nb_dev = DEV_RATE * len(files)

m = re.compile(r'\\([0-9]*)\\')

for f in files:
    # finding the label
    label = m.search(f).group(0)

    if nb_test >= 0: # test dataset
        path = TEST_PATH + "/" + label
        os.makedirs(path, exist_ok=True)
        copyfile(f, path + "/" + os.path.basename(f))
        nb_test -= 1
    elif nb_dev >= 0: # dev dataset
        path = DEV_PATH + "/" + label
        os.makedirs(path, exist_ok=True)
        copyfile(f, path + "/" + os.path.basename(f))
        nb_dev -= 1
    else: # train dataset
        path = TRAIN_PATH + "/" + label
        os.makedirs(path, exist_ok=True)
        copyfile(f, path + "/" + os.path.basename(f))

    

