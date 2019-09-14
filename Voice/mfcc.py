#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#! pip install librosa


# In[3]:


import librosa


# In[16]:


data, f = librosa.load("data/tsujita_a.mp3")
data2, f2 = librosa.load("data/kimura_a.mp3")


# In[17]:


mfcc=librosa.feature.mfcc(data)
mfcc2 = librosa.feature.mfcc(data2)


# In[6]:


data


# In[7]:


len(data)


# In[8]:


print(type(data))


# In[9]:


data.shape


# In[10]:


f


# In[11]:


mfcc


# In[12]:


from librosa import display


# In[ ]:





# In[13]:


plt.figure(figsize=(16,9))
librosa.display.waveplot(data, f)
plt.show()


# In[15]:


mfcc.shape


# In[18]:


mfcc2.shape


# In[ ]:




