#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import warnings
import math
import itertools
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from numpy import array
from keras.models import Sequential
# from keras.layers import LSTM
from keras.layers.convolutional import Conv1D    
from keras.layers import LSTM,Dense, Dropout, Activation, Bidirectional, Masking
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from numpy.random import seed
import os
print(os.listdir('../Public/Downloads/BTP/'))


# In[3]:


pip install pandas_datareader


# In[37]:


from pandas_datareader import data,wb


# In[38]:


import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


#training data

df1 = pd.read_csv("../Public/Downloads/BTP/2018-2019 -2021 training data.csv")


# In[47]:


#Validation data 

dfv=pd.read_csv("../Public/Downloads/BTP/jan-march 2022.csv")


# In[48]:


#Testing data

dft=pd.read_csv("../Public/Downloads/BTP/march-sept 2022.csv")


# In[62]:


df1.replace(to_replace=[None,"None"], value=np.nan, inplace=True)


# In[63]:


dft.replace(to_replace=[None,"None"], value=np.nan, inplace=True)


# In[64]:


dfv.replace(to_replace=[None,"None"], value=np.nan, inplace=True)


# In[67]:


dfv


# In[68]:


df1


# In[69]:


dft


# In[ ]:




