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


svc=SVC(kernel='linear')#モデル作成（本当はこっからkernel.gamma,cの調整が必要）


# In[4]:


row_data=np.empty((0, 22050), float)
f=np.zeros(30)
#空のarrayを作成
print(type(row_data))


# In[5]:


for i in range(8):
    my_data, my_f=librosa.load("data\kimura_out_a00{}.wav".format(i+2))
    row_data = np.append(row_data, np.array([my_data]), axis = 0)
    #まず木村さんの2-9番目の音声データ読み込み
# print(row_data[1])
# print(row_data[1].shape)
print(type(row_data))
print(row_data.shape)
row_data
# print(row_data)
#np.array(row_data)


# In[6]:


#実行は1回だけ！
for i in range(8,24):
    my_data,my_f=librosa.load("data\kimura_out_a0{}.wav".format(i+2))
    row_data = np.append(row_data, np.array([my_data]), axis = 0)
    #10-25番目の音声データ読み込み
print(type(f))
f
print(row_data.shape)


# In[11]:


mfccs=np.empty((20, 44, 24), float)
#mfccを入れるための空のarrayを作成
print(type(mfccs))
print(mfccs.shape)
mfccs


# In[12]:


for i in range(24):
    my_mfcc=librosa.feature.mfcc(row_data[i])
    mfccs = np.append(mfccs, np.array([my_mfcc]), axis = 0)
    print(my_mfcc.shape)
    
# #20個のmfccを入れる。二つ下のセルでなぜ２０個か説明します
# mfcc = librosa.feature.mfcc(data[0])
# print(type(mfcc))
# mfcc.shape


# In[ ]:


type(data[0])
#dataの一つ目の要素の型を確認（ndarray出ないといけないから)


# In[ ]:


for i in range(30):
    print(i,type(data[i]))
    #結局、全部の型を確認。２３個は大丈夫なので、きりよく20こにしました


# In[ ]:


data1=[None]*30
f1=[None]*30


# In[ ]:


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


# In[ ]:


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


mfcc2[0]
#確認したけどやっぱり違う
#その後、辻田さんの指摘で修正されました
#ここから先の見通しは、二つのmfcc群を20個ずつxにいれ、それに対応する正解ｙ（木村さん：０、辻田さん：１）みたいな感じで作ってSVCで学習する感じだと思います


# In[ ]:




