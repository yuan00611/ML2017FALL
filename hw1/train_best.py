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

data = pd.read_csv('PM25.csv', sep = ',',index_col = 'TimeLine')


# In[2]:

print(data.head())


# In[3]:

PM25 = np.array(data['PM2.5'])
PM10 = np.array(data['PM10'])
CH4 = np.array(data['CH4'])
CO = np.array(data['CO'])
NMHC = np.array(data['NMHC'])
NO = np.array(data['NO'])
NO2 = np.array(data['NO2'])
NOx = np.array(data['NOx'])
O3 = np.array(data['O3'])
SO2 = np.array(data['SO2'])
THC = np.array(data['THC'])
WIND_Direc = np.array(data['WIND_DIREC'])
rainfall = np.array(data['RAINFALL'])
feature_list = []

PM25predict_list = []


# In[4]:

for i in range(12):
    for j in range(471):
        a = PM25[480 * i + j: 480 * i + j + 9]
        b = PM10[480 * i + j: 480 * i + j + 9]
        c = CH4[480 * i + j: 480 * i + j + 9]
        d = CO[480 * i + j: 480 * i + j + 9]
        e = NMHC[480 * i + j: 480 * i + j + 9]
        f = NO[480 * i + j: 480 * i + j + 9]
        g = NO2[480 * i + j: 480 * i + j + 9]
        h = NOx[480 * i + j: 480 * i + j + 9]
        k = O3[480 * i + j: 480 * i + j + 9]
        l = SO2[480 * i + j: 480 * i + j + 9]
        m = THC[480 * i + j: 480 * i + j + 9]
        #n = WIND_Direc[480 * i + j: 480 * i + j + 9]
        #o = rainfall[480 * i + j: 480 * i + j + 9]
        ray = np.hstack((a, b, c, d, e, f, g, h, k, l, m))
        feature_list.append(ray)
        PM25predict_list.append(PM25[480 * i + j + 9])


# In[5]:

PM25_x = np.array(feature_list)
PM25_y = np.array(PM25predict_list)

PM25_x = np.concatenate((PM25_x, PM25_x ** 2), axis = 1)

#mms = StandardScaler()
#PM25_x = mms.fit_transform(PM25_x.astype(float))

PM25_x = np.concatenate((np.ones((PM25_x.shape[0],1)),PM25_x), axis=1)
print(PM25_x)
print(PM25_x.shape)
print(PM25_y)
print(PM25_y.shape)


# In[22]:

w = np.zeros(len(PM25_x[0]))
lr = 10
iteration = 10000


# In[23]:

w = np.matmul(np.matmul(inv(np.matmul(PM25_x.transpose(),PM25_x)),PM25_x.transpose()),PM25_y)
w


# In[24]:

hypo = np.dot(PM25_x,w)
loss = hypo - PM25_y
cost = np.sum(loss**2) / len(PM25_x)
cost_a  = math.sqrt(cost)
cost_a


# In[25]:

y_pred = []
for n in range(4700, 5652):
    score = 0
    for m in range(198):
        score = score + w[m + 1] * PM25_x[n, m + 1]
    y_p = score + w[0]
    y_pred.append(y_p)
y_test = PM25_y[4700:5652]

#RMSE
RMSE = 0
for o in range(942):
    RMSE = RMSE + (y_test[o] - y_pred[o]) ** 2
RMSE = math.sqrt(RMSE / len(y_pred))
RMSE


# In[26]:

plt.style.use('seaborn-bright')
fig = plt.figure(figsize=(20,8))

ax = plt.gca()
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=20)

z = [g for g in range(952)]
y1 = y_pred
y2 = PM25_y[4700:5652]
#plt.scatter(x=z, y=y, c='None')

plt.plot(z, y1, c='b',lw=2)
plt.plot(z, y2, c='r',lw=2)
plt.show()


# In[27]:

# save model
np.save('model_best.npy',w)