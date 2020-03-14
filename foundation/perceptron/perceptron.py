import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import random


def set_figsize(figsize=(3.5, 2.5)):
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def xyplot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.numpy(), y_vals.numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()

x = tf.Variable(tf.range(-8,8,0.1),dtype=tf.float32)
y = tf.nn.relu(x)

#relu激活函数
# #relu图
# xyplot(x, y, 'relu')
#
# #relu导数图
# with tf.GradientTape() as t:
#     t.watch(x)
#     y=y = tf.nn.relu(x)
# dy_dx = t.gradient(y, x)
# xyplot(x, dy_dx, 'grad of relu')

# #sigmoid激活函数
# y = tf.nn.sigmoid(x)
# xyplot(x, y, 'sigmoid')
#
# with tf.GradientTape() as t:
#     t.watch(x)
#     y=y = tf.nn.sigmoid(x)
# dy_dx = t.gradient(y, x)
# xyplot(x, dy_dx, 'grad of sigmoid')

# # tanh激活函数
# y = tf.nn.tanh(x)
# xyplot(x, y, 'tanh')
#
# with tf.GradientTape() as t:
#     t.watch(x)
#     y=y = tf.nn.tanh(x)
# dy_dx = t.gradient(y, x)
# xyplot(x, dy_dx, 'grad of tanh')

#1、数据加载

from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
batch_size = 256
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
x_train = x_train/255.0
x_test = x_test/255.0
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


############################################
#          多层感知机的从零开始实现           #
############################################
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#2、模型搭建
W_1=tf.Variable(tf.random.normal([num_inputs,num_hiddens]),name="weight_1")
b_1=tf.Variable(tf.zeros([num_hiddens]),name="bias_1")
W_2=tf.Variable(tf.random.normal([num_hiddens,num_outputs]),name="weight_2")
b_2=tf.Variable(tf.zeros([num_outputs]),name="bias_2")

#激活函数
def relu(x):
    return tf.math.maximum(x,0)

#模型搭建
def perceptron(x):
    x=tf.reshape(x,[-1,num_inputs])
    x=tf.matmul(x,W_1)+b_1
    x=relu(x)
    x=tf.matmul(x,W_2)+b_2
    x=tf.math.softmax(x)

    return x

#定义损失函数
def loss_fn(logits,labels):
    return tf.losses.sparse_categorical_crossentropy(y_true=labels,y_pred=logits)

def train(net, train_iter, test_iter, loss_fn, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_loss,train_acc,n=0.0,0.0,0
        for X,y in train_iter:
            with tf.GradientTape() as tape:
                pred=net(X)
                loss=tf.reduce_mean(loss_fn(pred,y))

            grad=tape.gradient(loss,params)
            if trainer is None:
                params[0].assign_sub(grad[0]*lr)
                params[1].assign_sub(grad[1] * lr)
                params[2].assign_sub(grad[2] * lr)
                params[3].assign_sub(grad[3] * lr)
            else:
                trainer.apply_gradients(zip(grad,params))

            train_loss+=loss.numpy()
            train_acc+=tf.reduce_sum(tf.cast(tf.argmax(pred,axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n+=y.shape[0]
        print("epoch: {} , loss: {},acc: {}".format(epoch,train_loss/n,train_acc/n))

num_epochs,lr=5,0.5
params=[W_1,b_1,W_2,b_2]
# train(perceptron,train_iter,test_iter,loss_fn,num_epochs,batch_size,params,lr)

############################################
#          多层感知机的从零开始实现           #
############################################
def tf_perceptron(num_inputs):
    inputs=tf.keras.layers.Input(shape=[28,28],name="input_layer")
    x=tf.keras.layers.Flatten()(inputs)
    x=tf.keras.layers.Dense(num_hiddens,activation="relu",name="hidden_layer")(x)
    x=tf.keras.layers.Dense(num_outputs,activation="softmax",name="output_layer")(x)
    model=tf.keras.Model(inputs=inputs,outputs=x)
    return model

tfmodel=tf_perceptron(num_inputs)
tfmodel.summary()

tfmodel.compile(loss=tf.losses.sparse_categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=lr),
                metrics=['acc'])
tfmodel.fit(x_train,y_train,epochs=num_epochs,validation_data=test_iter)

