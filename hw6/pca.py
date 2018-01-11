
# coding: utf-8

# In[1]:

from skimage import transform
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# In[2]:

def load_img(path):
    matrix = []
    for i in range(415):
        img = io.imread(path + str(i) + '.jpg')
        img = img.flatten()
        matrix.append(img)
    matrix = np.array(matrix)
    return matrix


# In[3]:

def draw_img(picture, path):
    picture = picture.reshape((600, 600, 3))
    picture -= np.min(picture)
    picture /= np.max(picture)
    picture = (picture * 255).astype(np.uint8)
    plt.imsave(path, picture)


# In[4]:

X_origin = load_img(sys.argv[1] + '/') #sys.argv[1]


# In[5]:

X_mean = np.mean(X_origin, axis = 1)
X = X_origin.T
U, s, V = np.linalg.svd(X - X_mean, full_matrices=False)


# In[9]:

x = str(sys.argv[2])
x = int(x[: -4])
weight = np.dot(X_origin[x], U[:,:4])
img = np.matmul(U[:,:4], weight)
img = img + X_mean[x]
draw_img(img, 'reconstruction.jpg')


# In[ ]:



