# coding: utf-8

# In[1]:

import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor

# In[2]:

x = pd.read_csv(sys.argv[1], sep = ',', header = 0)
del x['fnlwgt']
#print(x.shape)
#print(x.head())
y = pd.read_csv(sys.argv[2],sep =',', header = 0)
#print(y.shape)
#print(y.head())

x_test = pd.read_csv(sys.argv[3], sep = ',', header = 0)
del x_test['fnlwgt']

x_train = x.values.astype(float)
y_train = y.values[:,0].astype(float)
x_test = np.array(x_test.values)

# In[3]:

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# In[4]:

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


# In[5]:

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))
    
    X_all, Y_all = _shuffle(X_all, Y_all)
    
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid


# In[6]:

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))


# In[7]:

def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return


# In[8]:

def train(X_all, Y_all, save_dir):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    
    # Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((105,))
    mu2 = np.zeros((105,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((105,105))
    sigma2 = np.zeros((105,105))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    N1 = cnt1
    N2 = cnt2

    print('=====Saving Param=====')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    param_dict = {'mu1':mu1, 'mu2':mu2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
    for key in sorted(param_dict):
        print('Saving %s' % key)
        np.savetxt(os.path.join(save_dir, ('%s' % key)), param_dict[key])
    
    print('=====Validating=====')
    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)

    return

# In[12]:

#normalize + 刪一個feature 
x_train, x_test = normalize(x_train, x_test)
train(x_train, y_train, 'parameters')

