
# coding: utf-8

# In[19]:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional, Masking
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras
from keras.callbacks import ModelCheckpoint
import h5py
from keras.models import load_model

import pandas as pd
import numpy as np
from math import log, floor
from random import shuffle
import sys

import matplotlib.pyplot as plt
np.random.seed(1310) # for reproducibility


# In[2]:

f = open(sys.argv[1], 'r', encoding = 'utf8')

label_data = []
for i in range(200001):
    line = f.readline()
    label_data.append(line)
f.close()
#print(len(label_data))
#print(label_data[0])
#while (f.readline() != ''):
    


# In[3]:

label_Xtrain = [] 
label_ytrain = []
for i in range(len(label_data)-1):
    rr = label_data[i].find('\n')
    label_Xtrain.append(label_data[i][10:rr])
    label_ytrain.append(int(label_data[i][0]))
#print(label_Xtrain[0])
#print(label_ytrain[0])
label_ytrain = np.array(label_ytrain)


# In[6]:

# build tiken 分詞
token = Tokenizer(num_words=10000)
token.fit_on_texts(label_Xtrain)
#print(token.document_count)
#token.word_index


# In[7]:

import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[8]:

# turn the word to number list
x_train_seq = token.texts_to_sequences(label_Xtrain)
#x_train_seq[0]


# In[9]:

# make same length of number list
label_Xtrain = sequence.pad_sequences(x_train_seq , maxlen=30)


# In[10]:

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()


# In[11]:

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# In[12]:

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[14]:

X_train, Y_train, X_val, Y_val = split_valid_set(label_Xtrain, label_ytrain, 0.9) 


# In[ ]:

model4 = load_model('model.hdf5')


# In[ ]:

f1 = open(sys.argv[2], 'r', encoding = 'utf8')
unlabel_data = []

while True :
    i = f1.readline()
    if i=='': break
    unlabel_data.append(i)
f1.close()


# In[ ]:

unlabel_X_seq = token.texts_to_sequences(unlabel_data)


# In[ ]:

unlabel_X = sequence.pad_sequences(unlabel_X_seq , maxlen=30)


# In[ ]:

y_classes = model4.predict(unlabel_X)

for i in range(len(y_classes)):
    if y_classes[i] > 0.8:
        y_classes[i] = 1
    elif y_classes[i] < 0.2:
        y_classes[i] = 0
y_classes


# In[ ]:

semi_Y = y_classes.squeeze()


# In[ ]:

semi_X = unlabel_X[index, :]
semi_Y = semi_Y[index]


# In[ ]:

X_train = np.concatenate((X_train, semi_X))
Y_train = np.concatenate((Y_train, semi_Y))


# In[15]:

model1 = Sequential()
model1.add(Embedding(output_dim = 100, input_dim = 50000, input_length = 30,  mask_zero=True))
model1.add(Dropout(0.3))
model1.add(Bidirectional(LSTM(64, return_sequences = True)))
model1.add(Dropout(0.2))
model1.add(Bidirectional(LSTM(64)))
model1.add(Dense(units = 64, activation = 'relu'))
model1.add(Dropout(0.2))
#output layers
model1.add(Dense(units=1, activation='sigmoid'))


# In[16]:

model1.summary()


# In[17]:

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:

filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
                keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, verbose=1, mode='auto'),]


train_history= model1.fit(X_train, Y_train, batch_size=256,
                         epochs=10, verbose=2, validation_data=(X_val, Y_val), callbacks = checkpointer)


# In[ ]:

show_train_history(train_history, 'loss', 'val_loss')
show_train_history(train_history, 'acc', 'val_acc')

