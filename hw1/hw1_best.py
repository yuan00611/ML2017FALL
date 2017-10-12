
# coding: utf-8

# In[1]:

import sys
import numpy as np 
import scipy
import pandas as pd 
import random
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import csv


# read model
w = np.load('model_best.npy')


# In[13]:

t = pd.read_csv(sys.argv[1], sep = ',')
t = t.replace({'NR':0.0})


# In[15]:

test_x = []

for i in range(240):
    a = t.iloc[8 + i * 18, 2:].astype(float) #PM2.5
    b = t.iloc[7 + i * 18, 2:].astype(float) #PM10
    c = t.iloc[0 + i * 18, 2:].astype(float) #CH4
    d = t.iloc[1 + i * 18, 2:].astype(float) #CO
    e = t.iloc[2 + i * 18, 2:].astype(float) #NMHC
    f = t.iloc[3 + i * 18, 2:].astype(float) #NO
    g = t.iloc[4 + i * 18, 2:].astype(float) #NO2
    h = t.iloc[5 + i * 18, 2:].astype(float) #NOx
    k = t.iloc[6 + i * 18, 2:].astype(float) #O3
    l = t.iloc[11 + i * 18, 2:].astype(float) #SO2
    m = t.iloc[12 + i * 18, 2:].astype(float) #THC
    line = np.hstack((a, b, c, d, e, f, g, h, k, l, m))
    test_x.append(line)
test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
'''
print(test_x)
print(type(test_x))
print(test_x.shape)
'''

# In[16]:

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

#filename = "predict_best.csv"
text = open(sys.argv[2], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


# In[ ]:



