import os
import sys
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers, callbacks, initializers
from keras.models import Sequential, Model, load_model

model_file = './result/model043-loss0.000-vloss0.000.h5'

# ステップ数
LEN_SEQ  = 80
# サイクルあたりのステップ数
STEPS_PER_CYCLE = 80
# 生成するサイクル数
NUM_OF_CYCLE = 50

def min_max_normalization(array, axis=None):
    min = array.min(axis=axis, keepdims=True)
    max = array.max(axis=axis, keepdims=True)
    result = (array-min)/(max-min)
    return result

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
    plt.savefig("sincos.png")

if __name__=='__main__':
    validation()