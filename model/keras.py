import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os

#在内存中生成模拟数据
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X为-1到1之间连续的100个浮点数
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
    return train_X, train_Y   #以生成器的方式返回

train_data = GenerateData()
batch_size=10
dataset = tf.data.Dataset.from_tensor_slices( (  train_data[0],train_data[1]) )
dataset = dataset.shuffle(1000).repeat().batch(batch_size) #将数据集乱序、重复、批次划分.

#直接使用model定义网络
def model():
    inputs = tf.keras.Input(shape=(1,))
    outputs= tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#继承Model类来搭建模型
class my_model(tf.keras.Model):
    def __init__(self):
        super(my_model,self).__init__()
        self.fc=tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x=self.fc(inputs)
        return x

def model1():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    # model.add(tf.keras.layers.Dense(1,batch_input_shape=(None,1)))
    # model.add(tf.keras.layers.Dense(1, input_dim=1))

    return model


model=model()
model.summary()#打印网络

# 选择损失函数和优化方法
model.compile(loss='mse',optimizer='sgd')
# model.compile(loss=tf.losses.mean_squared_error,optimizer=tf.keras.optimizers.SGD)

#使用test_on_batch
for step in range(200):
    cost=model.train_on_batch(train_data[0],train_data[1])
    if step%10==0:
        print("loss",cost)

# #直接使用fit来训练
model.fit(x=train_data[0],y=train_data[1],batch_size=batch_size,epochs=20)

#直接使用Model类创建的网络
w,b=model.get_weights()
print("weights:",w)
print("bias:",b)
#直接使用Sequential的网络类创建
# w,b=model.layers[0].get_weights()
# print("weights:",w)
# print("bias:",b)

#保存及加载模型
model.save('my_model.h5')
# del model  #删除当前模型

cost = model.evaluate(train_data[0], train_data[1], batch_size = 10)#测试
print ('test loss: ', cost)

a = model.predict(train_data[0], batch_size = 10)#预测
print(a[:10])
print(train_data[1][:10])

del model  #删除当前模型
#加载
model = tf.keras.models.load_model('my_model.h5')

a = model.predict(train_data[0], batch_size = 10)
print("加载后的测试",a[:10])