#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
#librosaライブラリのインポート（初めての人はインストール必要かもです）


# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import SVC


# In[3]:


svc=SVC(kernel='rbf', gamma=0.1, C=10)#モデル作成（本当はこっからkernel.gamma,cの調整が必要）


# In[4]:


row_data_kimura=np.empty((0, 22050), float)
row_data_tsujita=np.empty((0, 22050), float)
#空のarrayを作成
print(type(row_data_kimura))


# In[5]:


for i in range(8):
    k_data, f=librosa.load(r"data\kimura_out_a00{}.wav".format(i+2))
    t_data, f=librosa.load(r"data\tsujita_out_a00{}.wav".format(i+2))
    row_data_kimura = np.append(row_data_kimura, np.array([k_data]), axis = 0)
    row_data_tsujita = np.append(row_data_tsujita, np.array([t_data]), axis = 0)
    #2-9番目の音声データ読み込み
# print(row_data[1])
# print(row_data[1].shape)
print(type(row_data_kimura))
print(row_data_kimura.shape)
print(row_data_tsujita.shape)
row_data_kimura
# print(row_data)
#np.array(row_data)


# In[ ]:





# In[6]:


#実行は1回だけ！
for i in range(8,24):
    k_data,f=librosa.load(r"data\kimura_out_a0{}.wav".format(i+2))
    t_data,f=librosa.load(r"data\tsujita_out_a0{}.wav".format(i+2))
    row_data_kimura = np.append(row_data_kimura, np.array([k_data]), axis = 0)
    row_data_tsujita = np.append(row_data_tsujita, np.array([t_data]), axis = 0)
    #10-25番目の音声データ読み込み
print(type(f))
print(row_data_kimura.shape)
print(f)


# In[13]:


mfccs_k_2=librosa.feature.mfcc(row_data_kimura[0])
mfccs_t_2=librosa.feature.mfcc(row_data_tsujita[0])
print(mfccs_k_2.shape)
print(mfccs_t_2.shape)
mfccs_k = mfccs_k_2[None,:,:]
mfccs_t = mfccs_t_2[None,:,:]
#mfccを入れるための空のarrayを作成(3次元に変換)
print(type(mfccs_k))
print(type(mfccs_t))
print(mfccs_k.shape)
mfccs_k


# In[ ]:





# In[14]:


for i in range(23):
    k_mfcc_2=librosa.feature.mfcc(row_data_kimura[i])
    t_mfcc_2=librosa.feature.mfcc(row_data_tsujita[i])
    k_mfcc=k_mfcc_2[None,:,:]
    t_mfcc=t_mfcc_2[None,:,:]
    #print(my_mfcc.shape)
    mfccs_k = np.concatenate([mfccs_k, k_mfcc], axis=0)
    mfccs_t = np.concatenate([mfccs_t, t_mfcc], axis=0)
    print(mfccs_k.shape)
    print(mfccs_t.shape)
# #23個のmfccを入れる。
# mfcc = librosa.feature.mfcc(data[0])
# print(type(mfcc))
# mfcc.shape


# In[29]:


y_train = np.ones(40)
y_test = np.ones(8)


# In[31]:


for i in range(20):
    y_train[i]=0
# 0:kimura, 1:tsujita

for i in range(4):
    y_test[i]=0


# In[18]:


import tensorflow as tf
import keras
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from keras.optimizers import SGD

from keras.utils import np_utils


# In[26]:


mfccs_k_train = mfccs_k[:20,:,:]
mfccs_t_train = mfccs_t[:20,:,:]
mfccs_k_test = mfccs_k[20:,:,:]
mfccs_t_test = mfccs_t[20:,:,:]
print(mfccs_k_train.shape)
print(mfccs_k_test.shape)


# In[33]:


mfccs_train = np.concatenate([mfccs_k_train, mfccs_t_train], axis=0)
mfccs_test = np.concatenate([mfccs_k_test, mfccs_t_test], axis=0)


# In[34]:


print(mfccs_train.shape)


# In[35]:


# 以上でデータの完成！


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


svc.fit(mfccs_k, y)


# In[ ]:


import tensorflow as tf
import keras
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from keras.optimizers import SGD

from keras.utils import np_utils


# In[ ]:


Y_train = np_utils.to_categorical(y, num_classes=2).astype('i')


# In[ ]:


batch_size=100


# In[ ]:


n_epoch=20


# In[ ]:




