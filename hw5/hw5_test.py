
# coding: utf-8

# In[24]:

import numpy as np
import pandas as pd


from keras.models import load_model

import pandas as pd
import numpy as np
from math import log, floor, sqrt
from random import shuffle
import csv
import sys

np.random.seed(1310) # for reproducibility


# In[5]:

model1 = load_model('Model1.hdf5')


# In[6]:

model2 = load_model('Model2.hdf5')


# In[7]:

model3 = load_model('Model3.hdf5')


# In[8]:

model4 = load_model('Model4.hdf5')


# # Test data

# In[9]:

test = pd.read_csv(sys.argv[1], header = 0, engine = 'python',
                  sep = ',', names = ['index', 'userid', 'movieid'])
#test.movieid = ratings.movieid.astype('category')
#test.userid = ratings.userid.astype('category')


# In[4]:

test_movieid = np.array(test.movieid)
test_userid = np.array(test.userid)


# In[12]:

def predict(model):
    y_class = model.predict([test_userid, test_movieid]) 
    y_class = (y_class * 1.116897661146206 + 3.5817120860388076)
    y_class[y_class < 1] = 1
    y_class[y_class > 5] = 5
    return y_class


# In[19]:

y_class1 = predict(model1)
y_class2 = predict(model2)
y_class3 = predict(model3)
y_class4 = predict(model4)


# In[21]:

d = np.hstack((y_class1, y_class2, y_class3, y_class4))


# In[22]:

y_classesall = np.mean(d, axis = 1)


# In[25]:

ans = []
for i in range(len(y_classesall)):
    ans.append([str(i+1)])
    ans[i].append(float(y_classesall[i]))

#filename = "predict_final.csv"
text = open(sys.argv[2], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["TestDataID","Rating"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


# In[ ]:



