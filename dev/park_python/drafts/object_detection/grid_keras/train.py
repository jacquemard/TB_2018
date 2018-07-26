##################################################################
#        MESURE DU TAUX D'OCCUPATION DE PARKINGS A L'AIDE        #
#                       DE CAMERAS VIDEOS                        #
# -------------------------------------------------------------- #
#               Rémi Jacquemard - TB 2018 - HEIG-VD              #
#                   remi.jacquemard@heig-vd.ch                   #
#                           July 2018                            #
# -------------------------------------------------------------- #
# Used to train the model, using a grid as output.               #
##################################################################

from pathlib import Path
import os
import sys
cur_path = Path(os.path.abspath(__file__) )
lib_path = str(cur_path.parent.parent.parent.parent.resolve())
sys.path.insert(0, lib_path)

from skimage import io
from skimage.viewer.viewers import ImageViewer
import os
import numpy as np
from pathlib import Path
import glob
import random
import math
#import model
from dataset_helper import pklot


HOME_PATH = str(Path.home())

TRAIN_DATASET = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_splitted/dev"
TEST_DATASET = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_splitted/test"
BASE_IMAGE = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_splitted/dev/2013-03-13_13_45_09.jpg"

LOG_PATH = "./keras_log"
CHECKPOINT_PATH = "./keras_checkpoints"

# IMAGE_SIZE = 480 x 270

GRID_SIZE = 22

image_template = io.imread(BASE_IMAGE)

# Used to transform each of the input images. Doing so, every images will be a bit different
"""
data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
"""


# loading data
def load_data(path):  
    images_config = pklot.images_with_xml(path)
    
    x = []
    y = []
    
    for image_config in images_config:
        image = io.imread(image_config[0])
        grid = pklot.get_cars_grid(image_template.shape[:2], GRID_SIZE, image_config[1])
        
        # wrapping value within an array to have correct model output shape
        grid = grid.reshape(grid.shape + (1,))

        x.append(image)
        y.append(grid)

    '''
    path = Path(path)
    for f in path.glob("*/*.bmp"):
        label = int(f.parts[-2])
        #print(label)
        image = io.imread(f)
        #print(image.shape)
        #images.append((image, label))
        x.append(image)
        y.append(label)
    
    #random.shuffle(images)
    #i = np.array(images)

    # normalizing labels
    y = y / np.max(y) 
    
    return (np.array(x), y)  #.reshape((len(y), 1))
    '''

    return (np.array(x), np.array(y))

x_train, y_train = load_data(TRAIN_DATASET)
print("x_shape: {}".format(x_train.shape))
print("y_shape: {}".format(y_train.shape))
#print("x--------------\n{}".format(x_train))
#print("y--------------\n{}".format(y_train))


# Defining model

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor


def model_func():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape=image_template.shape, data_format="channels_last", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), use_bias=False))  
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    print(model.output_shape)
    #model.add(Dropout(0.2))
    #model.add(Flatten())
    #model.add(Dense(200))
    #model.add(Activation("relu"))
    #model.add(Dropout(0.2))

    # using padding (padding = 'same') is inconsistent with keras and tensorflow 
    # with odd kernel / even input, wich was the case here 
    # (see https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
    # it seems to that when a low padding is necessary, it is simply ignored by keras
    # so, the input has been padded manually
    
    height = math.ceil(model.output_shape[1]/GRID_SIZE)
    print(height)
    print(model.output_shape)
    print(GRID_SIZE)
    width = math.ceil(model.output_shape[2]/GRID_SIZE)
    
    top_pad = 0
    left_pad = 0
    bottom_pad = (height * GRID_SIZE) - model.output_shape[1]
    right_pad = (width * GRID_SIZE) - model.output_shape[2]

    print(bottom_pad, " ", right_pad)

    model.add(ZeroPadding2D(padding=((top_pad, bottom_pad), (left_pad, right_pad))))
    model.add(Conv2D(1, kernel_size=(height, width), strides=(height, width), padding='valid'))
    model.add(Activation("sigmoid"))

    """
    height = math.floor(model.output_shape[1]/GRID_SIZE)
    width = math.floor(model.output_shape[2]/GRID_SIZE)
    model.add(Conv2D(1, kernel_size=(height, width), strides=(height, width), padding='valid'))
    """

    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.summary()

    print("model compiled")
    return model

# making different images
# just moving slightly within the image
data_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    validation_split=0.2)

data_generator.fit(x_train)
train_generator = data_generator.flow(x_train, y_train, batch_size=4, subset='training')
test_generator = data_generator.flow(x_train, y_train, batch_size=4, subset='validation')

# callbacks
tb_call_back = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(CHECKPOINT_PATH + "/{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)

# fitting
m = model_func()
#m.fit_generator(train_generator, epochs=50, callbacks=[tb_call_back, checkpoint], validation_data=test_generator, validation_steps=50)
#m.fit_generator(train_generator, epochs=50, callbacks=[tb_call_back, checkpoint], validation_data=test_generator, validation_steps=50)

#m.fit(x_train, y_train, epochs=50, callbacks=[tb_call_back, checkpoint], validation_split=0.2)
#m.predict(x_train)

#m.save_weights('pklot_reg_1.h5')

'''
data_generator = ImageDataGenerator()
train_generator = data_generator.flow_from_directory(TRAIN_DATASET, target_size=image_template.shape[0:2], batch_size=4, color_mode="rgb")
test_generator = data_generator.flow_from_directory(TEST_DATASET, target_size=image_template.shape[0:2], batch_size=4, color_mode="rgb")
'''
#regressor = KerasRegressor(model_func, epochs=20)
#kfold = KFold(n_splits=3)
#results = cross_val_score(regressor, x_train, y_train)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#regressor.fit(x_train, y_train)
#print(regressor.predict(x_train))

# fitting

'''
tb_call_back = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)

print(len(train_generator))
model.fit_generator(train_generator,
    steps_per_epoch=None,
    epochs=1,
    verbose=2,
    callbacks=[tb_call_back]
)
'''
# Saving the model
'''
model.save_weights('pklot_1.h5')

model.evaluate_generator(test_generator)
'''
