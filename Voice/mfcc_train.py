#!/usr/bin/env python
# coding: utf-8

# In[8]:


import librosa
#librosaライブラリのインポート（初めての人はインストール必要かもです）


# In[9]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import SVC


# In[10]:


svc=SVC(kernel='rbf', gamma=0.1, C=10)#モデル作成（本当はこっからkernel.gamma,cの調整が必要）


# In[11]:


row_data_kimura=np.empty((0, 22050), float)
row_data_tsujita=np.empty((0, 22050), float)
#空のarrayを作成
print(type(row_data_kimura))


# In[30]:


for i in range(8):
    k_data, f=librosa.load("data\kimura_out_a00{}.wav".format(i+2))
    t_data, f=librosa.load("data\tsujita_out_a00{}.wav".format(i+2))
    row_data_kimura = np.append(row_data_kimura, np.array([k_data]), axis = 0)
    #2-9番目の音声データ読み込み
# print(row_data[1])
# print(row_data[1].shape)
print(type(row_data_kimura))
print(row_data_kimura.shape)
row_data_kimura
# print(row_data)
#np.array(row_data)


# In[ ]:





# In[13]:


#実行は1回だけ！
for i in range(8,24):
    k_data,f=librosa.load("data\kimura_out_a0{}.wav".format(i+2))
    row_data_kimura = np.append(row_data_kimura, np.array([k_data]), axis = 0)
    #10-25番目の音声データ読み込み
print(type(f))
print(row_data_kimura.shape)
print(f)


# In[14]:


mfccs_k_2=librosa.feature.mfcc(row_data_kimura[0])
print(mfccs_k_2.shape)
mfccs_k = mfccs_k_2[None,:,:]
#mfccを入れるための空のarrayを作成(3次元に変換)
print(type(mfccs_k))
print(mfccs_k.shape)
mfccs_k


# In[ ]:





# In[15]:


for i in range(23):
    k_mfcc_2=librosa.feature.mfcc(row_data_kimura[i])
    k_mfcc=k_mfcc_2[None,:,:]
    #print(my_mfcc.shape)
    mfccs_k = np.concatenate([mfccs_k, k_mfcc], axis=0)
    print(mfccs_k.shape)
    
# #23個のmfccを入れる。
# mfcc = librosa.feature.mfcc(data[0])
# print(type(mfcc))
# mfcc.shape


# In[16]:


y = np.ones(23)


# In[17]:


y


# In[18]:


data1=[None]*30
f1=[None]*30


# In[19]:


for i in range(8):
    data1[i],f1[i]=librosa.load(r"C:\Users\ttaki\workshop-repository-e\Voice\data\tsujita_out_a00{}.wav".format(i+2),sr=44100)


# In[ ]:


for i in range(8,23):
    data1[i],f1[i]=librosa.load(r"C:\Users\ttaki\workshop-repository-e\Voice\data\tsujita_out_a0{}.wav".format(i+2),sr=44100)


# In[ ]:


mfcc2=[None]*30


# In[ ]:


for i in range(30):
    print(i,type(data1[i]))


# In[20]:


for i in range(20):
    mfcc2[i]=librosa.feature.mfcc(data1[i])
    #ここまで木村さんの音声の時と同様に、辻田さんの方でもやってます


# In[ ]:


for i in range(20):
    print('{},{}'.format(mfcc[i].shape,mfcc2[i].shape))
    #念のため、両方のmfccの型を確認したところ違っている！？


# In[ ]:


mfcc2[0].shape


# In[ ]:


mfcc[0]


# In[ ]:


mfcc2[0]#確認したけどやっぱり違う
#その後、辻田さんの指摘で修正されました
#ここから先の見通しは、二つのmfcc群を20個ずつxにいれ、それに対応する正解ｙ（木村さん：０、辻田さん：１）みたいな感じで作ってSVCで学習する感じだと思います


# In[16]:


svc.fit(mfccs_k, y)


# In[21]:


import tensorflow as tf
import keras
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from keras.optimizers import SGD

from keras.utils import np_utils


# In[22]:


Y_train = np_utils.to_categorical(y, num_classes=2).astype('i')


# In[23]:


batch_size=100


# In[24]:


n_epoch=20


# In[ ]:




