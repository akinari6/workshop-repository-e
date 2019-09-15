#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
#librosaライブラリのインポート（初めての人はインストール必要かもです）


# In[3]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import SVC


# In[5]:


svc=SVC(kernel='linear')#モデル作成（本当はこっからkernel.gamma,cの調整が必要）


# In[7]:


data=np.array([None]*30)
f=np.array([None]*30)
#空のarrayを作成
print(type(data))


# In[9]:


for i in range(7):
    data[i],f[i]=librosa.load("data\kimura_out_a00{}.wav".format(i+2),sr=44100)
    #まず木村さんの2-9番目の音声データ読み込み
    #データのパスを自分のパソコンで合わせてしまっているので、直してください。ごめんなさい
print(type(data[1]))
data[1].shape


# In[16]:


for i in range(8,23):
    data[i],f[i]=librosa.load(r"C:\Users\ttaki\workshop-repository-e\Voice\data\kimura_out_a0{}.wav".format(i+2),sr=44100)
    #10-25番目の音声データ読み込み


# In[17]:


mfcc=[None]*30
#mfccを入れるための空のリストを作成


# In[22]:


for i in range(20):
    mfcc[i]=librosa.feature.mfcc(data[i])
    
    #20個のmfccを入れる。二つ下のセルでなぜ２０個か説明します


# In[19]:


type(data[0])
#dataの一つ目の要素の型を確認（ndarray出ないといけないから)


# In[24]:


for i in range(30):
    print(i,type(data[i]))
    #結局、全部の型を確認。２３個は大丈夫なので、きりよく20こにしました


# In[26]:


data1=[None]*30
f1=[None]*30


# In[32]:


for i in range(8):
    data1[i],f1[i]=librosa.load(r"C:\Users\ttaki\workshop-repository-e\Voice\data\tsujita_out_a00{}.wav".format(i+2),sr=44100)


# In[29]:


for i in range(8,23):
    data1[i],f1[i]=librosa.load(r"C:\Users\ttaki\workshop-repository-e\Voice\data\tsujita_out_a0{}.wav".format(i+2),sr=44100)


# In[40]:


mfcc2=[None]*30


# In[33]:


for i in range(30):
    print(i,type(data1[i]))


# In[41]:


for i in range(20):
    mfcc2[i]=librosa.feature.mfcc(data1[i])
    #ここまで木村さんの音声の時と同様に、辻田さんの方でもやってます


# In[42]:


for i in range(20):
    print('{},{}'.format(mfcc[i].shape,mfcc2[i].shape))
    #念のため、両方のmfccの型を確認したところ違っている！？


# In[36]:


mfcc2[0].shape


# In[37]:


mfcc[0]


# In[38]:


mfcc2[0]
#確認したけどやっぱり違う
#その後、辻田さんの指摘で修正されました
#ここから先の見通しは、二つのmfcc群を20個ずつxにいれ、それに対応する正解ｙ（木村さん：０、辻田さん：１）みたいな感じで作ってSVCで学習する感じだと思います


# In[ ]:




