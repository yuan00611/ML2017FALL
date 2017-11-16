
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from math import log, floor
from random import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import regularizers
import csv
import matplotlib.pyplot as plt
import sys


# In[2]:

data = pd.read_csv(sys.argv[1], sep = ',', header = 0)


# In[3]:

x_list = []
for i in range(len(data)):
    a = np.fromstring(data['feature'][i], dtype=float, sep=" ")
    x_list.append(a)


# In[4]:

x_train = np.array(x_list) / 255
x_train = np.clip(x_train, 1e-8, 1-(1e-8))


# In[5]:

l = pd.get_dummies(data, columns = ['label'])
del l['feature']
y_train = np.array(l)


# In[6]:

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# In[7]:

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[8]:

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()


# In[9]:

X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y_train, 0.95)


# # CNN

# In[10]:

X_train4D = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32')
X_valid4D = X_valid.reshape(X_valid.shape[0], 48, 48, 1).astype('float32')


# In[17]:

np.random.seed(1310) # for reproducibility
model1 = Sequential()
model1.add(Conv2D(filters = 64,
                  kernel_size = (3, 3),
                  padding = 'valid',
                  input_shape = (48, 48, 1)))
model1.add(LeakyReLU(alpha=.1))
model1.add(BatchNormalization())
model1.add(Conv2D(filters = 128,
                  kernel_size = (3, 3),
                  padding = 'valid'))
model1.add(LeakyReLU(alpha=.1))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Conv2D(filters = 256,
                  kernel_size = (3, 3),
                  padding = 'valid'))
model1.add(LeakyReLU(alpha=.1))
model1.add(BatchNormalization())
model1.add(Conv2D(filters = 256,
                  kernel_size = (3, 3),
                  padding = 'valid'))
model1.add(LeakyReLU(alpha=.1))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Conv2D(filters = 512,
                  kernel_size = (3, 3),
                  padding = 'valid'))
model1.add(LeakyReLU(alpha=.1))
model1.add(BatchNormalization())
model1.add(Conv2D(filters = 512,
                  kernel_size = (3, 3),
                  padding = 'valid'))
model1.add(LeakyReLU(alpha=.1))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2, 2)))

model1.add(Flatten())

model1.add(Dense(1024))
model1.add(LeakyReLU(alpha=.1))
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
model1.add(Dense(1024))
model1.add(LeakyReLU(alpha=.1))
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
model1.add(Dense(7, activation = 'softmax'))


# In[18]:

print(model1.summary())


# In[19]:

model1.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[20]:

from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import ModelCheckpoint
import h5py

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=60,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train4D)


# In[46]:

filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'),]


train_history =  model1.fit_generator(datagen.flow(X_train4D, Y_train, batch_size=256),
                    epochs = 300,
                    validation_data=(X_valid4D, Y_valid),
                    samples_per_epoch=X_train4D.shape[0], verbose = 2,
                    callbacks = checkpointer)


# In[34]:

train = model1.evaluate(X_train4D, Y_train)
print('\nloss: ', train[0])
print('Test acc: ', train[1], '\n')

result = model1.evaluate(X_valid4D, Y_valid)
print('\nloss: ', result[0])
print('Test acc: ', result[1])


# In[35]:

show_train_history(train_history, 'loss', 'val_loss')
show_train_history(train_history, 'acc', 'val_acc')


