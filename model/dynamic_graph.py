import numpy as np
import matplotlib.pyplot as plt

#（1）生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

import tensorflow as tf


# 定义学习参数
W = tf.Variable(tf.random.uniform([1]),dtype=tf.float32, name="weight")
b = tf.Variable(tf.zeros([1]),dtype=tf.float32, name="bias")

def getcost(x,y):#定义函数，计算loss值
    # 前向结构
    z = tf.cast(tf.multiply(np.asarray(x,dtype = np.float32), W)+ b,dtype = tf.float32)
    cost =tf.reduce_mean( tf.square(y - z))#loss值
    return cost

def grad( inputs, targets):#获取模型参数的梯度。
    with tf.GradientTape() as tape:
        # tape.watch([W,b])
        loss_value = getcost(inputs, targets)
    return tape.gradient(loss_value,[W,b])

# 随机梯度下降法作为优化器
learning_rate = 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

training_epochs = 20  #迭代训练次数
display_step = 2

plotdata = { "batchsize":[], "loss":[] }#收集训练参数

step=0
for epoch in range(training_epochs):
    for (x, y) in zip(train_X, train_Y):
        step+=1
        grads = grad(x, y)
        optimizer.apply_gradients(zip(grads, [W, b]))
        # 显示训练中的详细信息
        if step % display_step == 0:
            cost = getcost(x, y)
            print("Epoch:", step , "cost=", cost.numpy(), "W=", W.numpy(), "b=", b.numpy())

            plotdata["batchsize"].append(step*len(train_X))
            plotdata["loss"].append(cost.numpy())

print ("cost=", getcost (train_X, train_Y).numpy() , "W=", W.numpy(), "b=", b.numpy())

#显示模型
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, W * train_X + b, label='Fitted line')
plt.legend()
plt.show()

def moving_average(a, w=10):#定义生成loss可视化的函数
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

plotdata["avgloss"] = moving_average(plotdata["loss"])
plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.show()