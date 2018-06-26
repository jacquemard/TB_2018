from skimage import io
from skimage.viewer.viewers import ImageViewer
import os
import numpy as np
from pathlib import Path
import glob
import random
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold

HOME_PATH = str(Path.home())

TRAIN_DATASET = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/train"
TEST_DATASET = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/test"
BASE_IMAGE = HOME_PATH + "/DS/PKLot/PKLot/UFPR05_processed_splitted/test/2/2013-02-22_06_05_00.bmp"

LOG_PATH = "./keras_log"

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

    #images = []
    
    x = []
    y = []

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
    
    return (np.array(x), np.array(y))  #.reshape((len(y), 1))
    
x_train, y_train = load_data(TRAIN_DATASET)
print(y_train)
print(x_train.shape)

# Defining model
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor

image_template = io.imread(BASE_IMAGE)


print(image_template.shape[0:2])

def model_func():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation="relu", input_shape=image_template.shape, data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    print("model compiled")
    return model

# fitting
tb_call_back = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)

m = model_func()
m.fit(x_train, y_train, epochs=50, callbacks=[tb_call_back], validation_split=0.2)
m.predict(x_train)

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
