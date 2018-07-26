##################################################################
#        MESURE DU TAUX D'OCCUPATION DE PARKINGS A L'AIDE        #
#                       DE CAMERAS VIDEOS                        #
# -------------------------------------------------------------- #
#              Rémi Jacquemard - TB 2018 - HEIG-VD               #
#                   remi.jacquemard@heig-vd.ch                   #
#               https://github.com/remij1/TB_2018                #
#                           July 2018                            #
# -------------------------------------------------------------- #
# First test creating a keras model, handling parking detection  #
# with regression.                                               #
##################################################################

from keras import Sequential
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from skimage import io
import os
import numpy as np
from pathlib import Path

HOME_PATH = str(Path.home())

TRAIN_DATASET = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/dev"
TEST_DATASET = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/test"
BASE_IMAGE = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/test/2/2013-02-22_06_05_00.bmp"

LOG_PATH = "./keras_log"
tb_call_back = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)

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

# custom loss function
def custom_loss(true_label, predicted_label):
    print(K.eval(true_label))
    # finding best label
    '''
    K.map_fn(lambda x: True if x == 1.0 else False, true_label)
    K.foldl(lambda (acc, x) : (acc + 1 if x == 1 else) )
    best = [i for i in true_label if i == 1][0]
    print(best)
    '''
    print(true_label, " ", predicted_label)
    return int(true_label) - int(predicted_label)

# Defining model
image_template = io.imread(BASE_IMAGE)

#print(image_template.shape[0:2])

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3), strides=(1, 1), activation="relu", input_shape=image_template.shape, data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(40, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(41, activation='softmax'))
model.compile(loss=custom_loss, optimizer='adam')
model.summary()

print("model compiled")
# generators
data_generator = ImageDataGenerator()
train_generator = data_generator.flow_from_directory(TRAIN_DATASET, target_size=image_template.shape[0:2], batch_size=4, color_mode="rgb")
test_generator = data_generator.flow_from_directory(TEST_DATASET, target_size=image_template.shape[0:2], batch_size=4, color_mode="rgb")



# fitting
print(len(train_generator))
model.fit_generator(test_generator,
    steps_per_epoch=16,
    epochs=20,
    verbose=1,
    callbacks=[tb_call_back]
)

# Saving the model
model.save_weights('pklot_1.h5')

scores = model.evaluate_generator(train_generator)
print(scores)
