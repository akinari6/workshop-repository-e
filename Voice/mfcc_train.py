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





# In[36]:


mfccs_train_max = mfccs_train.max()


# In[37]:


mfccs_test_max = mfccs_test.max()


# In[38]:


mfccs_max = max(mfccs_train_max, mfccs_test_max)


# In[40]:


#正規化
mfccs_train /= mfccs_max
mfccs_test /= mfccs_max


# In[ ]:


# 以上でデータの完成！


# In[ ]:


#ニューラルネットワークにぶち込む


# In[ ]:


import tensorflow as tf
import keras
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from keras.optimizers import SGD

from keras.utils import np_utils


# In[41]:


Y_train = np_utils.to_categorical(y_train, num_classes=2).astype('i')
Y_test = np_utils.to_categorical(y_test, num_classes=2).astype('i')


# In[46]:


batch_size=100


# In[47]:


n_epoch=20


# In[42]:


model = Sequential()  # モデルのインスタンスを作成（モデルを作る度に作成する必要がある）
# addメソッドで層を追加していく。
# Flatten: 二次元配列を一次元配列に変換する層
# 入力層に配置しているときはinput_shapeに入力サイズを指定。
model.add(Flatten(input_shape=(20, 44)))

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


# In[43]:


# 損失関数(＝誤差関数）は分類では定番の交差エントロピー誤差（回帰ならRMSEが定番）
# 最適化手法は基本的な確率的勾配降下法(SGD)（一番ポピュラーなのはAdam　＊後述）
# 評価方法は精度（Accuracy）に指定。学習時での出力評価に使われる。

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy']) 


# In[48]:


#実際に学習させる（注意：hist変数は学習終了後生成されるので中断したら参照できない）
hist = model.fit(mfccs_train,
                 Y_train,
                 epochs=n_epoch,
                 validation_data=(mfccs_test, Y_test),
                 verbose=1,
                 batch_size=batch_size)


# In[50]:


# 学習精度をテストデータで確認できる（学習時にvalidation_dataとして入力していたら必要ない）
loss_and_metrics = model.evaluate(mfccs_test, Y_test)
loss_and_metrics #（損失値、精度）を返す


# In[51]:


hist.history


# In[52]:


# プロットして損失値と精度の推移を視覚化するのは非常に重要な作業なので特に理由がなければ毎度表示する。
# 過学習や学習不足が確認できる。


# 損失値(Loss)の遷移のプロット
def plot_history_loss(hist):
    
    # hist.historyに辞書型で損失値や精度が入っているので取得して表示
    plt.plot(hist.history['loss'],label="loss for training")
    plt.plot(hist.history['val_loss'],label="loss for validation")
    
    #matplotlibの細かい設定
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    
    plt.show()

    
# 精度(Accuracy)の遷移のプロット
def plot_history_acc(hist):
    plt.plot(hist.history['acc'],label="loss for training")
    plt.plot(hist.history['val_acc'],label="loss for validation")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.show()

plot_history_loss(hist)
plot_history_acc(hist)


# In[54]:


# testデータ内のサンプルをピックアップ
# indexを指定して任意の１秒間の音声を選択
index = 3
plt.imshow(mfccs_test[index])
plt.show()


# In[55]:


# 予測を行い､答え合わせをする

# 予測はクラスの確率的表現（softmax関数より）で出てくるのでargmaxで最大の列番号を取得
pred = model.predict(mfccs_test[index].reshape(1, 20, 44)).argmax()
ans  = Y_test[index].argmax()

print('predict: ', pred)
print('answer : ', ans)

if pred == ans:
    print('正解です｡')
else:
    print('不正解です')


# In[56]:


# 混同行列を出力
# testデータに対して行うことに注意
from sklearn.metrics import confusion_matrix as cm
result = model.predict(mfccs_test).argmax(axis=1)
cm(y_test, result)  # y_testはOne-Hot表現にする前のデータ形式に注意


# In[57]:


result


# In[62]:


# 混同行列をグラフで出力する関数
def plot_cm(y_true, y_pred):
    confmat = cm(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xticks(np.arange(0, 1, 1)) # x軸の目盛りを指定
    plt.yticks(np.arange(0, 1, 1)) # y軸の目盛りを指定
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()


# In[63]:


# 混同行列をグラフで出力
# testデータに対して行うことに注意
plot_cm(y_test, result)


# In[ ]:


#全部木村さんの音声じゃ

