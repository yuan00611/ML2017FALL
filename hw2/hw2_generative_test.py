# coding: utf-8

# In[1]:

import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor

x = pd.read_csv(sys.argv[1], sep = ',', header = 0)
del x['fnlwgt']

x_train = x.values.astype(float)

# In[2]:
x_test = pd.read_csv(sys.argv[2], sep = ',', header = 0)
del x_test['fnlwgt']

x_test = np.array(x_test.values)

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


# In[3]:

def infer(X_test, save_dir, output_dir):
    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    mu1 = np.loadtxt(os.path.join(save_dir, 'mu1'))
    mu2 = np.loadtxt(os.path.join(save_dir, 'mu2'))
    shared_sigma = np.loadtxt(os.path.join(save_dir, 'shared_sigma'))
    N1 = np.loadtxt(os.path.join(save_dir, 'N1'))
    N2 = np.loadtxt(os.path.join(save_dir, 'N2'))

    # Predict
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    # Write output
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

# In[13]:
x_train, x_test = normalize(x_train, x_test)
infer(x_test, 'parameters', sys.argv[3])