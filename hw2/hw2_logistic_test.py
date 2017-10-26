
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

x_test = pd.read_csv(sys.argv[2], sep = ',', header = 0)
del x_test['fnlwgt']
del x_test[' Prof-school']
del x_test[' Adm-clerical']
del x_test[' Greece']
del x_test[' Trinadad&Tobago']
del x_test[' United-States']
del x_test[' Vietnam']

x_train = x.values.astype(float)
x_test = np.array(x_test.values)

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


# In[27]:

def infer(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'prediction.csv')
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

x_train, x_test = normalize(x_train, x_test)
infer(x_test, 'parameters', sys.argv[3])