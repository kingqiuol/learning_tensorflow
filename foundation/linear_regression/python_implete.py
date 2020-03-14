import tensorflow as tf
from time import time

a=tf.ones((1000,))
b=tf.ones((1000,))

# 逐一做标量加法
start = time()
c = tf.Variable(tf.zeros((1000,)))
for i in range(1000):
    c[i].assign(a[i] + b[i])
print("逐一做标量加法:",time() - start)

# 直接做矢量加法
start = time()
c.assign(a + b)
print("直接做矢量加法",time() - start)

#线性回归模型从零开始的实现
import tensorflow as tf
print(tf.__version__)
from matplotlib import pyplot as plt
import random

# 1、生成数据集
#设训练数据集样本数为1000，输入个数（特征数）为2。
# 给定随机生成的批量样本特征 X∈R1000×2，我们使用线性回归模型真实权重 w=[2,−3.4]⊤ 和偏差 b=4.2，
# 以及一个随机噪声项 ϵ 来生成标签y=Xw+b+ϵ
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal((num_examples, num_inputs),stddev = 1)
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += tf.random.normal(labels.shape,stddev=0.01)

print(features[0], labels[0])

def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize

# set_figsize()
# plt.scatter(features[:, 1], labels, 1)
# plt.show()

#2、读取数据
batch_size=5
dataset=tf.data.Dataset.from_tensor_slices((features,labels))
dataset=dataset.shuffle(1000).batch(batch_size=batch_size)

# for x,y in dataset:
#     print(x,y)

#3、初始化模型参数
W=tf.Variable(tf.random.normal((2,1)),name="weights")
b=tf.Variable(tf.zeros((1)),name="bias")
print(W,b)

#4、定义模型
def linear(x,W,b):
    return tf.matmul(x,W)+b

#5、定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 /2

#6、定义优化器
def sgd(params, lr, batch_size, grads):
    """Mini-batch stochastic gradient descent."""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)

lr = 0.03
num_epochs = 3
net = linear
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in dataset:
        with tf.GradientTape() as t:
            t.watch([W,b])
            l = loss(net(X, W, b), y)
        grads = t.gradient(l, [W, b])
        sgd([W, b], lr, batch_size, grads)
    train_l = loss(net(features, W, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))

# true_w = [2, -3.4]
# true_b = 4.2
print(W,b)

def model(input_dim):
    inputs=tf.keras.layers.Input(shape=(1,input_dim))
    outs=tf.keras.layers.Dense(1,
                               kernel_initializer=tf.initializers.RandomNormal(stddev=0.01),
                               name="fc")(inputs)
    model=tf.keras.Model(inputs=inputs,outputs=outs)
    return model

model=model(2)
model.summary()

loss_fn=tf.keras.losses.MeanSquaredError()
optimizer=tf.keras.optimizers.SGD(lr=0.03)

num_epochs=3
for epoch in range(num_epochs):
    for X, y in dataset:
        with tf.GradientTape() as tape:
            logits=model(X)
            loss=loss_fn(logits,y)

        grad=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))

    train_l = loss_fn(model(features), labels)
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))

# true_w = [2, -3.4]
# true_b = 4.2
print(model.get_weights())
