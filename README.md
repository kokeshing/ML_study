# 環境構築

## Google colaboratoryの準備

- [Google colaboratory](https://colab.research.google.com/)にアクセス
- 右下のノートブックを新規作成をクリック
- Python3の新しいノートブックを選択
- 上メニューのランタイムからランタイムのタイプを変更をクリック
- ハードウェアアクセラレータにGPUを選択して保存

## Kerasをインストール

今回はKerasを使います。理由としては

- 書き方がTensorflowに比べ簡単（まあTensorflowもいつかやるけどな）
- Tensorflowに吸収されているので応用が効く
- Kerasで慣れてからTensorflow使うと良いと思います
- 正直ハイパラパラチューニングするのでもめんどくさいのにモデルきちんと書くのめんどくさい
- まあKerasで物足りないなと思うときはくる（はず）
- 時間がない（一番大きい）

以上のようになります。

今回は時間がない（未来視）のでコピペして動くようにしてあります。
ノートブックの一番上のコードセルに

```
!mkdir ./result
!pip install -q keras
import keras
```
を入力して[Shift+Enter]またはセルの左の再生ボタンをクリックして下さい

```
Using TensorFlow backend.
```

と出力されれば正しくimportされています。

今回はLSTMを用いてsin関数（時間あったらもうちょっと複雑なやつも）の予測をさせてみましょう。

LSTMってどうなってんだよ（哲学）と気になる方もいますでしょうが説明する時間がないです（たぶん）

噛み砕きまくって説明するとLSTMはRNNの一種であり時系列データ（ex: 波動、為替、動画、などなど）のタスク
に対してよい効力を発揮します。

時間あったらこのへんも説明する😃。

## LSTMの学習

### 必要なライブラリ類をimport

```
import os
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers, callbacks, initializers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, LSTM
```

### 定数を定義

```
# 隠れ層の数
N_HIDDEN = 300
# ステップ数
LEN_SEQ  = 80
# 特徴の次元数
IN_NEUR  = 1
# バッチサイズ
BATCH_NM = 256
# エポック数
EPOCH_NM = 1000
# サイクルあたりのステップ数
STEPS_PER_CYCLE = 80
# 生成するサイクル数
NUM_OF_CYCLE = 50

result_dir = './result/'
```

### LSTMモデルを構築する関数を定義

```
def create_lstm_model():
    input_layer = Input(shape=(LEN_SEQ, IN_NEUR))
    lstm_out = LSTM(N_HIDDEN, input_shape=(LEN_SEQ, IN_NEUR), bias_initializer=initializers.Constant(value=(np.random.rand() / 10.0)), return_sequences=False)(input_layer)
    model_output = Dense(1, activation='linear')(lstm_out)
    model = Model(input_layer, model_output)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.RMSprop())
    return model
```

### sin関数を一定サイクル生成してそれを分割して訓練データを作成

```
def mk_sin_traindata():
    fx_data= [math.sin(x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE)]

    x_data = []
    y_data = []
    for i in range(0, len(fx_data)-LEN_SEQ):
        x_data.append(np.array(fx_data[i:i+LEN_SEQ]).reshape(LEN_SEQ, 1))
        y_data.append(fx_data[i+LEN_SEQ])

    if len(x_data) == len(y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        p = np.random.permutation(len(x_data))
        x_data = x_data[p]
        y_data = y_data[p]

        return x_data, y_data
```

### callbackの定義（Kerasではepochが終わるごとに関数を呼び出せる）

今回は

- モデルをepoch毎に自動的に保存してくれる関数
- モデルの精度が伸びなくなったら学習を途中で打ち切る関数

を呼び出す

```
cp_cb = callbacks.ModelCheckpoint(
    filepath = './result/model{epoch:03d}-loss{loss:.3f}-vloss{val_loss:.3f}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto')

es_cb = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=0,
    mode='auto')
```

### モデルとデータを生成して学習を開始

```
np.random.seed(0)
model = create_lstm_model()

x_train , y_train = mk_traindata()

model.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_NM,
    epochs=EPOCH_NM,
    verbose=1,
    callbacks=[cp_cb, es_cb],
    validation_split=0.3,
    shuffle=False)
```

## 検証

いい感じに学習ができたら

```
!ls ./result
```

を実行してval_lossが一番小さいモデルのファイル名をコピーしてから
下の検証用関数を定義

```
model_file = "./result/コピーしたファイル名"

def validation():
    model = load_model(os.path.join(model_file))
    x = [i for i in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE+ 200)]

    # 検証用の答え
    val_y = [math.sin(x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE+ 200)]
    val_y = val_y[-200:]

    predict = [math.sin(x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE)]
    for i in range(0, 200):
        data = np.array([np.array(predict[-LEN_SEQ:]).reshape(LEN_SEQ, 1)])

        pred = model.predict(data)[0]

        predict = np.append(predict, pred)

    predict = predict[-200:]
    x = x[-200:]

    plt.plot(x, val_y, color="red")
    plt.plot(x, predict, color="blue")
    plt.show
```

検証用関数を実行

今回は200ステップ出力させてみる

赤色が正解、青色が今回学習したモデルでの結果


```
validation()
```

## めんどくせえよ関数とか定数部分最初に全部コピーさせろやという需要

### モデル・学習部分

```
!mkdir ./result
!pip install -q keras

import os
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers, callbacks, initializers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, LSTM

# 隠れ層の数
N_HIDDEN = 300
# ステップ数
LEN_SEQ  = 80
# 特徴の次元数
IN_NEUR  = 1
# バッチサイズ
BATCH_NM = 256
# エポック数
EPOCH_NM = 1000
# サイクルあたりのステップ数
STEPS_PER_CYCLE = 80
# 生成するサイクル数
NUM_OF_CYCLE = 50

result_dir = './result/'

def create_lstm_model():
    input_layer = Input(shape=(LEN_SEQ, IN_NEUR))
    lstm_out = LSTM(N_HIDDEN, input_shape=(LEN_SEQ, IN_NEUR), bias_initializer=initializers.Constant(value=(np.random.rand() / 10.0)), return_sequences=False)(input_layer)
    model_output = Dense(1, activation='linear')(lstm_out)
    model = Model(input_layer, model_output)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.RMSprop())
    return model

def mk_traindata():
    fx_data= [math.sin(x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE)]

    x_data = []
    y_data = []
    for i in range(0, len(fx_data)-LEN_SEQ):
        x_data.append(np.array(fx_data[i:i+LEN_SEQ]).reshape(LEN_SEQ, 1))
        y_data.append(fx_data[i+LEN_SEQ])

    if len(x_data) == len(y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        p = np.random.permutation(len(x_data))
        x_data = x_data[p]
        y_data = y_data[p]

        return x_data, y_data

np.random.seed(0)
model = create_lstm_model()

cp_cb = callbacks.ModelCheckpoint(
    filepath = './result/model{epoch:03d}-loss{loss:.3f}-vloss{val_loss:.3f}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto')

es_cb = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=0,
    mode='auto')

x_train , y_train = mk_traindata()

model.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_NM,
    epochs=EPOCH_NM,
    verbose=1,
    callbacks=[cp_cb, es_cb],
    validation_split=0.3,
    shuffle=False)
```

### 検証用関数の部分

```
model_file = "./result/コピーしたファイル名"

def validation():
    model = load_model(os.path.join(model_file))
    x = [i for i in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE+ 200)]

    # 検証用の答え
    val_y = [math.sin(x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE+ 200)]
    val_y = val_y[-200:]

    predict = [math.sin(x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE)]
    for i in range(0, 200):
        data = np.array([np.array(predict[-LEN_SEQ:]).reshape(LEN_SEQ, 1)])

        pred = model.predict(data)[0]

        predict = np.append(predict, pred)

    predict = predict[-200:]
    x = x[-200:]

    plt.plot(x, val_y, color="red")
    plt.plot(x, predict, color="blue")
    plt.show
```

```
validation()
```

## おまけ

時間余ったらもうちょっと複雑な関数も学習させてみよう（適当）

新しいノートブックを開いてtraindata()とvalidation()とサイクル数だけを下のデータに変える

これを実行し終わったらfit()してモデル生成してからvalidation()を実行するだけ

```
# サイクルあたりのステップ数
STEPS_PER_CYCLE = 80
# 生成するサイクル数
NUM_OF_CYCLE = 200

def mk_traindata():
    fx_data= [3 * math.sin(5 * x * (2 * math.pi / STEPS_PER_CYCLE)) - 5 * math.cos(3 * x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE)]

    x_data = []
    y_data = []
    for i in range(0, len(fx_data)-LEN_SEQ):
        x_data.append(np.array(fx_data[i:i+LEN_SEQ]).reshape(LEN_SEQ, 1))
        y_data.append(fx_data[i+LEN_SEQ])

    if len(x_data) == len(y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        p = np.random.permutation(len(x_data))
        x_data = x_data[p]
        y_data = y_data[p]

        return x_data, y_data

def validation():
    model = load_model(os.path.join(model_file))
    x = [i for i in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE+ 200)]

    # 検証用の答え
    val_y = [3 * math.sin(5 * x * (2 * math.pi / STEPS_PER_CYCLE)) - 5 * math.cos(3 * x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE+ 200)]
    val_y = val_y[-200:]

    predict = [3 * math.sin(5 * x * (2 * math.pi / STEPS_PER_CYCLE)) - 5 * math.cos(3 * x * (2 * math.pi / STEPS_PER_CYCLE)) for x in range(0, NUM_OF_CYCLE * STEPS_PER_CYCLE)]
    for i in range(0, 200):
        data = np.array([np.array(predict[-LEN_SEQ:]).reshape(LEN_SEQ, 1)])

        pred = model.predict(data)[0]
        if i < 100:
            print(pred)

        predict = np.append(predict, pred)

    predict = predict[-200:]
    x = x[-200:]

    plt.plot(x, val_y, color="red")
    plt.plot(x, predict, color="blue")
    plt.show
```

## 結果

うまくいけばこんな感じな出力が得られるはず

### sin波

![sin_rmsprop](https://user-images.githubusercontent.com/33972190/40005483-fa0be122-57d2-11e8-8884-872595820be5.png)

### 3sin5x-5cos3x

![sincos_cycle200](https://user-images.githubusercontent.com/33972190/40005486-fc66d0c6-57d2-11e8-9198-e035ddeb30c9.png)