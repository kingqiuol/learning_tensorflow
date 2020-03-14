import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

#1、数据加载
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(len(x_train),len(x_test))
#
# feature,label=x_train[0],y_train[0]
# print(feature.shape, feature.dtype)
# print(label, type(label), label.dtype)
#
# def get_fashion_mnist_labels(labels):
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
#
# #显示
# def show_fashion_mnist(images, labels):
#     _, figs = plt.subplots(1, len(images), figsize=(12, 12))
#     for f, img, lbl in zip(figs, images, labels):
#         f.imshow(img.reshape((28, 28)))
#         f.set_title(lbl)
#         f.axes.get_xaxis().set_visible(False)
#         f.axes.get_yaxis().set_visible(False)
#     plt.show()
#
# X, y = [], []
# for i in range(10):
#     X.append(x_train[i])
#     y.append(y_train[i])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

#2、数据加载
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4

batch_size=16
x_train = tf.cast(x_train, tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test,tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))

############################################
#        softmax回归的从零开始实现            #
############################################
#3、定义模型
input_dim=784
output_dim=10
W=tf.Variable(tf.random.normal((input_dim,output_dim),stddev=0.01),name="weight")
b=tf.Variable(tf.zeros([output_dim]),name="bias")

def softmax(x):
    return tf.exp(x)/tf.reduce_mean(tf.exp(x),axis=1,keepdims=True)

def model(x):
    x=tf.reshape(x,[-1,W.shape[0]])
    out=tf.matmul(x,W)+b
    return softmax(out)

#3、定义损失函数和优化器
def loss_fn(logits,labels):
    labels=tf.cast(tf.reshape(labels,[-1,1]),dtype=tf.int32)
    labels=tf.one_hot(labels,depth=logits.shape[-1])
    labels = tf.cast(tf.reshape(labels, shape=[-1, logits.shape[-1]]), dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(logits,labels)+1e-8)

#准确率
def acc_fn(preds,labels):
    labels=tf.cast(labels,tf.int64)
    preds=tf.cast(tf.argmax(preds,axis=1),tf.int64)
    return tf.reduce_mean(tf.equal(preds,labels))

#测试集上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y,dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n

#4、模型训练
def train(net, train_iter, test_iter, loss_fn, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_loss, train_acc, n = 0.0, 0.0, 0
        for step,(X,y) in enumerate(train_iter):
            with tf.GradientTape() as tape:
                out=net(X)
                loss=loss_fn(out,y)
                loss=tf.reduce_sum(loss)
            grads=tape.gradient(loss,params)
            if trainer is None:
                sample_grads = grads
                params[0].assign_sub(grads[0] * lr)
                params[1].assign_sub(grads[1] * lr)
            else:
                trainer.apply_gradients(zip(grads, params))  # “softmax回归的简洁实现”一节将用到

            train_loss+=loss.numpy()
            train_acc+=tf.reduce_sum(tf.cast(tf.argmax(out,axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n+=y.shape[0]
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss / n, train_acc / n, test_acc))

num_epochs, lr = 5, 1e-3
# trainer = tf.keras.optimizers.SGD(lr)
# train(model, train_iter, test_iter, loss_fn, num_epochs, batch_size, [W, b], lr)

############################################
#           softmax回归的简洁实现            #
############################################
def tf_model(num_class):
    inputs=tf.keras.layers.Input(shape=[28,28])
    x=tf.keras.layers.Flatten()(inputs)
    x=tf.keras.layers.Dense(num_class,activation='softmax',name="fc")(x)
    model=tf.keras.Model(inputs=inputs,outputs=x)
    return model
tfmodel=tf_model(num_class=output_dim)
tfmodel.summary()

optimier=tf.keras.optimizers.Adam(lr=lr)
tfmodel.compile(optimizer=tf.keras.optimizers.Adam(0.003),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

tfmodel.fit(x_train,y_train,epochs=5,batch_size=256,validation_data=test_iter)


if sigmoid_noise>0:
    noise=random_ops.random_normal(array_ops.shape(score),dytpe=score.type,seed=seed)
    score+=sigmoid_noise*nosie
