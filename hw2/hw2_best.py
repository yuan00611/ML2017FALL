
# coding: utf-8

# In[1]:

import csv
import numpy as np
import pandas as pd
from numpy.linalg import inv
import random
import math
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# In[2]:

x = pd.read_csv(sys.argv[1], sep = ',', header = 0)
y = pd.read_csv(sys.argv[2],sep =',', header = 0)

del x['fnlwgt']
del x[' Prof-school']
del x[' Adm-clerical']
del x[' Greece']
del x[' Trinadad&Tobago']
del x[' United-States']
del x[' Vietnam']

#print(x.shape)
#print(x.head())

#print(y.shape)
#print(y.head())
x_array = x.values.astype(float)
y_array = y.values[:,0].astype(float)

# In[3]:

x_test = pd.read_csv(sys.argv[3], sep = ',', header = 0)

del x_test['fnlwgt']
del x_test[' Prof-school']
del x_test[' Adm-clerical']
del x_test[' Greece']
del x_test[' Trinadad&Tobago']
del x_test[' United-States']
del x_test[' Vietnam']

#print(x_test.shape)
#print(x_test.head())

x_test = np.array(x_test.values.astype(float))


#print(x_test.shape)
#print(type(x_test))



# In[6]:

# add bias
x_array = np.concatenate((np.ones((x_array.shape[0],1)),x_array), axis=1)
x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test), axis=1)


# In[7]:

X_train, X_test, y_train, y_test = train_test_split(x_array, y_array, test_size = 0.2, random_state = 42)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)


# In[8]:

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[9]:

print('Accuracy of logictic regression classifier on test set:{:.5f}.'.format(classifier.score(X_test, y_test)))


# In[10]:

y_pred = classifier.predict(x_test)


# In[11]:

ans = []
for i in range(len(x_test)):
    ans.append([str(i+1)])
    a = int(y_pred[i])
    ans[i].append(a)

filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


