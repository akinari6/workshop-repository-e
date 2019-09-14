#!/usr/bin/env python
# coding: utf-8

# In[1]:



#  「あ～～～～～～～～」識別システム

#  息の続く限り「あ～～」と録音してできたMP3ファイルを用いて学習させ、
#  発声者が誰なのかを識別するシステム
#  (なぜこれをするのかは議事録（9/10）を参照)


# In[2]:



# 学習用＆テスト用データの作成

# １、パソコンのボイスレコーダーか何かを使って「あ～～」とできるだけ長く吹き込み、
# 'yourname_a'という名前で保存する。

# ２、おそらく上で作られたファイルは.m4aファイルになってしまうので、これを
# .mp3に変換する（変換サイトが早い https://online-audio-converter.com/ja/)

# ３、


# In[3]:


#まずはimportから
from pydub import AudioSegment #これが音声データ解析のためのパッケージ
                               #pydubの他にffmpegもインストールする必要がある
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


tsujita_a = AudioSegment.from_file('data/tsujita_a.mp3', 'mp3') #ファイルの取得
kimura_a = AudioSegment.from_file('data/kimura_a.mp3', 'mp3')

frames_per_second_t = tsujita_a.frame_rate #元のサンプルレート
frames_per_second_k = kimura_a.frame_rate 
frames_per_second_t, frames_per_second_k #4.4 kHzだった


# In[5]:


duration_seconds_t = tsujita_a.duration_seconds #録音時間
duration_seconds_k = kimura_a.duration_seconds
duration_seconds_t, duration_seconds_k #30 sec 位


# In[6]:


channels_t = tsujita_a.channels # 1:mono, 2:streo
channels_k = kimura_a.channels
channels_t, channels_k #どっちもstreo→片方のみ取り出す必要あり


# In[7]:


#左右のマイクのデータが交互に入っていると思われるので１つおきに取り出せば
#ステレオからモノに変換できる

tsujita_a_data2 = np.array(tsujita_a.get_array_of_samples()) #streo
tsujita_a_data = tsujita_a_data2[::2] #mono
kimura_a_data2 = np.array(kimura_a.get_array_of_samples())
kimura_a_data = kimura_a_data2[::2] 


# In[8]:


plt.plot(tsujita_a_data[600000:601000])  #辻田の「あ」


# In[9]:


plt.plot(kimura_a_data[600000:601000]) #木村さんの「あ」


# In[10]:


#木村さんの声の周波数が最も高いので、これをフーリエ変換して、最高周波数を求める

sample_for_max_freq = kimura_a_data[600000:601000] #上の切り出したデータ
spec = np.fft.fft(sample_for_max_freq) #フーリエ変換


# In[11]:


abs_spec = np.abs(spec) #スペクトルの周波数のリスト（単位はHzでなくカウント数のままなので注意）
abs_spec


# In[12]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.yscale('log') #log スケール
ax.hist(abs_spec, bins = 50)
fig.show()


# In[13]:


#300000カウントはおかしい・・・
#調べたところによると、人の声の周波数は大きくて1000 Hz位→4.4 kカウント位がマックスのはず
#静かなところでやったはずなんだけど・・・とりあえずはカットせずに行く
#1000カウントで３周期分は入るし・・・まあなんとかなりそう


# In[14]:


plt.plot(tsujita_a_data) #辻田の録音データ　使えそうな場所をさがす


# In[15]:


plt.plot(kimura_a_data) #木村さんの録音データ


# In[16]:


tsujita_length = len(tsujita_a_data) #ベクトルの要素数
kimura_length = len(kimura_a_data)
tsujita_length, kimura_length


# In[46]:


X = np.empty((0, 1000), float) #音声データ
y = np.ones(2000, dtype=int) #結果 0:辻田、1:木村


# In[47]:


# 1データは1,000カウント分、100,000～1,100,000までの1,000,000カウント分が使えるから合計1000データ
for i in range(1000):
    datum_t = np.array([tsujita_a_data[100000+1000*i : 101000+1000*i]]) #1000個のデータ配列をXに追加
    X = np.append(X, datum_t, axis=0)
for i in range(1000):
    datum_k = np.array([kimura_a_data[100000+1000*i : 101000+1000*i]])
    X = np.append(X, datum_k, axis=0)
#Xには2000個の音声データ配列が入っている

# 参考：https://qiita.com/fist0/items/d0779ff861356dafaf95


# In[48]:


y[:1000] = 0#初めの1000個は0（辻田の音声であることを示す）


# In[53]:


print(X.shape)
print(y.shape)
print(type(X))
print(type(y))
print(y)


# In[ ]:


# 2次元配列を3次元配列に変換


# In[54]:


# データができたので学習させる。
# ニューラルネットワークを使ってみる


# In[55]:


# ! pip install -U tensorflow==1.6.0


# In[56]:


# ! pip install -U keras==2.0


# In[57]:


import tensorflow as tf
import keras

# 層構造のモデルを定義するためのメソッド（kerasのモデル構築で必ず使う）
from keras.models import Sequential

# Denseは層の生成メソッド、Activationは活性化関数を定義するためのメソッド
# Flattenは二次元配列を一次元配列に変換する層
from keras.layers import Dense, Activation, Flatten

# SGD : ディープラーニングにおいて最も基本的な最適化手法
from keras.optimizers import SGD

#one-hot-encoding用のライブラリ
from keras.utils import np_utils


# In[64]:


# データをtrain用とtest用に分ける
X_train = X[:1700]
X_test = X[1700:]
y_train = y[:1700]
Y_test = y[1700:] #クラスラベルが一つしかないので実質one-hot表現になっている

print(X_train.shape)
print(X_test.shape)
print(y_train)
print(Y_test)


# In[38]:


# 数値がでかいので0.0～1.0に正規化(必要？？？)
X_max = X.max()
X_train /= X_max
X_test /= X_max


# In[61]:


# 結果をone-hot表現に変換
Y_train = np_utils.to_categorical(y_train, num_classes=2).astype('i')


# In[65]:


print(Y_train.shape)
print(Y_test.shape)


# In[66]:


Y_train[0] #一応確認


# In[67]:


# ニューラルネットワークの実装


# ミニバッチに含まれるサンプル数を指定
# データからサンプルを100個ずつ取り出して学習する（経験則的だが分類するクラス数以上にすると良いことが多い）
batch_size = 50

# epoch数を指定（計算リソースと相談、過学習が起きることもあるので多ければ良いというわけではないが基本は多い方が精度が高い）
n_epoch = 20


# In[76]:


#-----------
# MLPモデル
#-----------

model = Sequential()  # モデルのインスタンスを作成（モデルを作る度に作成する必要がある）
# addメソッドで層を追加していく。
# Flatten: 二次元配列を一次元配列に変換する層
# 入力層に配置しているときはinput_shapeに入力サイズを指定。
model.add(Flatten(input_shape=(1700,1000)))

# Dense: 全結合（線形結合）レイヤーです。引数に出力サイズ（1次元）を指定する。
model.add(Dense(900))

# Activation: 活性化関数を定義。今回は最もポピュラーなReLU関数。他にも"sigmoid"：シグモイド関数などがある。
model.add(Activation('relu'))

# 以下同じような要領で層を重ねていく
model.add(Dense(1000))
model.add(Activation('relu'))

model.add(Dense(500))
model.add(Activation('relu'))

# 出力層：最後の線形結合レイヤーは分類するクラス数に指定。
# 活性化関数は分類なのでSoftmax関数（回帰なら恒等関数＝そのまま出力）
model.add(Dense(2))
model.add(Activation('softmax'))


# In[77]:


# 損失関数(＝誤差関数）は分類では定番の交差エントロピー誤差（回帰ならRMSEが定番）
# 最適化手法は基本的な確率的勾配降下法(SGD)（一番ポピュラーなのはAdam　＊後述）
# 評価方法は精度（Accuracy）に指定。学習時での出力評価に使われる。

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])  


# In[72]:


#実際に学習させる（注意：hist変数は学習終了後生成されるので中断したら参照できない）
hist = model.fit(X_train,
                 Y_train,
                 epochs=n_epoch,
                 validation_data=(X_test, Y_test),
                 verbose=1,
                 batch_size=batch_size)


# In[ ]:




