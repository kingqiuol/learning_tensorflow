import tensorflow as tf
from tensorflow import keras
import time
import math
import numpy as np
import sys
sys.path.append("..")
from rnn import load_data_jay_lyrics,RNNModel,train_and_predict_rnn,train_and_predict_rnn_keras

#读取数据集
(corpus_indices, char_to_idx, idx_to_char,vocab_size) = load_data_jay_lyrics()

#初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

def get_params():
    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32))

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32))

    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    # 附上梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    return params

#定义模型
def init_gru_state(batch_size, num_hiddens):
    return (tf.zeros(shape=(batch_size, num_hiddens)), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X=tf.reshape(X,[-1,W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        R = tf.sigmoid(tf.matmul(X, W_xr) + tf.matmul(H, W_hr) + b_r)
        H_tilda = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

# 训练模型并创作歌词

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)

gru_layer = keras.layers.GRU(num_hiddens,time_major=True,return_sequences=True,return_state=True)
model = RNNModel(gru_layer, vocab_size)
train_and_predict_rnn_keras(model, num_hiddens, vocab_size,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)



