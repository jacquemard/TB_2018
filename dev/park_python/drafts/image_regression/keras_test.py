from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os
import numpy as np
from pathlib import Path

HOME_PATH = str(Path.home())

TRAIN_DATASET = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/dev"
TEST_DATASET = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/test"
BASE_IMAGE = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/test/2/2013-02-22_06_05_00.bmp"

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

# Defining model
image_template = io.imread(BASE_IMAGE)

print(image_template.shape[0:2])

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation="relu", input_shape=image_template.shape, data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(41, activation='softmax'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')

print("model compiled")
# generators
data_generator = ImageDataGenerator()
train_generator = data_generator.flow_from_directory(TRAIN_DATASET, target_size=image_template.shape[0:2], color_mode="rgb")
test_generator = data_generator.flow_from_directory(TEST_DATASET, target_size=image_template.shape[0:2], color_mode="rgb")


# fitting
model.fit_generator(train_generator,
    steps_per_epoch=200,
    epochs=20,
    verbose=2,
    validation_data=test_generator,
    validation_steps=50
)

# Saving the model
model.save_weights('pklot_1.h5')

model.evaluate_generator(test_generator)