
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sys

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[3]:

def take_part_data(length=500):
    # only take part of data to try 
    raw_image = np.load(sys.argv[1])
    part_data = raw_image[:length]
    return part_data
 


# In[4]:

X = take_part_data(140000)


# In[5]:

X = X.astype('float32') / 255.


# In[11]:

pca = PCA(copy = True, iterated_power = 'auto', n_components = 280, svd_solver = 'randomized', tol = 0.0, whiten = True)
pca.fit(X)


# In[12]:

components = pca.transform(X)


# In[13]:

kmeans = KMeans(n_clusters=2, random_state=0).fit(components)


# In[14]:

exam = pd.read_csv(sys.argv[2], sep = ',' , header = 0)

IDs, idx1, idx2 = np.array(exam.iloc[:,0]),np.array(exam.iloc[:,1]),np.array(exam.iloc[:,2])


# In[22]:

o = open(sys.argv[3], 'w')
o.write('ID,Ans\n')
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1
    else:
        pred = 0
    o.write('{},{}\n'.format(idx, pred))
o.close()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



