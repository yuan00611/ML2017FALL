
# coding: utf-8

# In[1]:

import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

# In[2]:

x = pd.read_csv(sys.argv[1], sep = ',', header = 0)
del x['fnlwgt']
del x[' Prof-school']
del x[' Adm-clerical']
del x[' Greece']
del x[' Trinadad&Tobago']
del x[' United-States']
del x[' Vietnam']
#print(x.shape)
#print(x.head())
y = pd.read_csv(sys.argv[2],sep =',', header = 0)
#print(y.shape)
#print(y.head())
x_test = pd.read_csv(sys.argv[3], sep = ',', header = 0)
del x_test['fnlwgt']
del x_test[' Prof-school']
del x_test[' Adm-clerical']
del x_test[' Greece']
del x_test[' Trinadad&Tobago']
del x_test[' United-States']
del x_test[' Vietnam']

x_train = x.values.astype(float)
y_train = y.values[:,0].astype(float)
x_test = np.array(x_test.values)

# In[21]:

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# In[22]:

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test


# In[23]:

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[24]:

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))


# In[25]:

def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return


# In[26]:

def train(X_all, Y_all, save_dir):

    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    # Initiallize parameter, hyperparameter
    w = np.zeros((99,))
    b = np.zeros((1,))
    l_rate = 0.1
    batch_size = 32
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 1001
    save_param_iter = 50

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        # Random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.sum(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.sum(-1 * (np.squeeze(Y) - y))

            # SGD updating parameters
            w = w - l_rate * w_grad
            b = b - l_rate * b_grad

    return

x_train, X_test = normalize(x_train, x_test)
train(x_train, y_train, 'parameters')