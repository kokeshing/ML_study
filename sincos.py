import os
import sys
import random
import math
import numpy as np
import pandas as pd
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

log_filepath = './log/'
result_dir = './result/'
model_file = './result/model012-loss0.017-vloss0.013.h5'

def min_max_normalization(array, axis=None):
    min = array.min(axis=axis, keepdims=True)
    max = array.max(axis=axis, keepdims=True)
    result = (array-min)/(max-min)
    return result

def create_lstm_model():
    input_layer = Input(shape=(LEN_SEQ, IN_NEUR))
    lstm_out = LSTM(N_HIDDEN, input_shape=(LEN_SEQ, IN_NEUR), bias_initializer=initializers.Constant(value=(np.random.rand() / 10.0)), return_sequences=False)(input_layer)
    model_output = Dense(1, activation='linear')(lstm_out)
    model = Model(input_layer, model_output)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.RMSprop())
    return model

# sin関数のデータを用意
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
    else:
        printf("ラベルとデータの長さが違います")
        sys.exit(1)

if __name__=='__main__':
    np.random.seed(0)
    model = create_lstm_model()

    cp_cb = callbacks.ModelCheckpoint(
        filepath = './result/model{epoch:03d}-loss{loss:.3f}-vloss{val_loss:.3f}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto')

    tb_cb = callbacks.TensorBoard(
        log_dir=log_filepath,
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    es_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=100,
        verbose=0,
        mode='auto')

    x_train , y_train = mk_traindata()

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_NM,
        epochs=EPOCH_NM,
        verbose=1,
        callbacks=[cp_cb, tb_cb, es_cb],
        validation_split=0.3,
        shuffle=False)

    model.save(os.path.join(result_dir, 'trained_model.h5'))