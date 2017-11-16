
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from keras.models import load_model
import csv
import sys


# In[45]:

model = load_model('hw3-Model.hdf5')


# In[5]:

testdata = pd.read_csv(sys.argv[1], header = 0, sep = ',')


# In[6]:

t_list = []
for i in range(len(testdata)):
    a = np.fromstring(testdata['feature'][i], dtype=float, sep=" ")
    t_list.append(a)


# In[7]:

x_test = np.array(t_list) / 255


# In[8]:

x_test4D = x_test.reshape(x_test.shape[0], 48, 48, 1).astype('float32')


# In[46]:

ans = model.predict(x_test4D)
y_classes = ans.argmax(axis=-1)


# In[47]:

ans = []
for i in range(len(y_classes)):
    ans.append([str(i)])
    ans[i].append(y_classes[i])

text = open(sys.argv[2], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()



