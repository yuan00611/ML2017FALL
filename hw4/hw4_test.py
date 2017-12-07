
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import csv
import sys


# In[2]:

model = load_model('model.hdf5')


# In[3]:

f = open(sys.argv[1], 'r', encoding = 'utf8')
f.readline()

test_data = []
for i in range(200001):
    line = f.readline()
    test_data.append(line)
f.close()
#len(test_data)


# In[4]:

label_Xtest = [] 
for i in range(len(test_data)-1):
    rr = test_data[i].find('\n')
    label_Xtest.append(test_data[i][2:rr])
#label_Xtest[0]


# In[5]:

import pickle

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[6]:

x_test_seq = tokenizer.texts_to_sequences(label_Xtest)

#x_test_seq[0]


# In[7]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

label_Xtest = sequence.pad_sequences(x_test_seq , maxlen=30)


# In[8]:

#label_Xtest


# In[9]:

y_classes = model.predict(label_Xtest)

for i in range(len(y_classes)):
    if y_classes[i] > 0.5:
        y_classes[i] = 1
    else:
        y_classes[i] = 0
#y_classes


# In[10]:

import csv
ans = []
for i in range(len(y_classes)):
    ans.append([str(i)])
    ans[i].append(int(y_classes[i]))

#filename = "predict_82.21.csv"
text = open(sys.argv[2], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


# In[ ]:



