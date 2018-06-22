import glob
import os
import re
import random
from shutil import copyfile

DATASET_PATH = "C:/DS/PKLot/PKLot/PKLot/UFPR05_processed"
OUTPUT_PATH = "C:/DS/PKLot/PKLot/PKLot/UFPR05_processed_splitted"

TRAIN_PATH = OUTPUT_PATH + "/train"
TEST_PATH = OUTPUT_PATH + "/test"

TRAIN_RATE = 0.8


#shuffling files
files = glob.glob(DATASET_PATH + "/*/*.bmp")
random.shuffle(files)

nb_train = TRAIN_RATE * len(files)

m = re.compile(r'\\([0-9]*)\\')

for f in files:
    # finding the label
    label = m.search(f).group(0)

    if nb_train >= 0: # train dataset
        path = TRAIN_PATH + "/" + label
        os.makedirs(path, exist_ok=True)
        copyfile(f, path + "/" + os.path.basename(f))
    else: #test dataset
        path = TEST_PATH + "/" + label
        os.makedirs(path, exist_ok=True)
        copyfile(f, path + "/" + os.path.basename(f))

    nb_train -= 1

