import numpy as np
import tensorflow as tf

def batch_norm(X,is_training,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not is_training:#如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        x_hat=(X-moving_mean).np.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape)==2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean=X.mean(axis=0)
            var=((X-mean)**2).mean(axis=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean=X.mean(axis=(0,2,3),keep_dim=True)
            var=((X-mean)**2).mean(axis=(0,2,3),keep_dim=True)

        # 训练模式下用当前的均值和方差做标准化
        x_hat=(X-mean)/np.sqrt(var+eps)

        #更新移动平均的均值和方差
        moving_mean=momentum*moving_mean+(1.0-momentum)*mean
        moving_var=momentum*moving_var+(1.0-momentum)*var
    Y=gamma*x_hat+beta
    return Y,moving_mean,moving_var

class BatchNormalization(tf.keras.layers.Layer):
    def __init__(self,decay=0.9, epsilon=1e-5, **kwargs):
        super(BatchNormalization,self).__init__()
        self.decay=decay
        self.epsilon=epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=[input_shape[-1], ],
                                     initializer=tf.initializers.ones,
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                     shape=[input_shape[-1], ],
                                     initializer=tf.initializers.zeros,
                                     trainable=True)
        self.moving_mean = self.add_weight(name='moving_mean',
                                     shape=[input_shape[-1], ],
                                     initializer=tf.initializers.zeros,
                                     trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
                                     shape=[input_shape[-1], ],
                                     initializer=tf.initializers.ones,
                                     trainable=False)
        super(BatchNormalization, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        """
        variable = variable * decay + value * (1 - decay)
        """
        delta = variable * self.decay + value * (1 - self.decay)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            batch_mean, batch_variance = tf.nn.moments(inputs, list(range(len(inputs.shape) - 1)))
            mean_update = self.assign_moving_average(self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = tf.nn.batch_normalization(inputs,
                                           mean=mean,
                                           variance=variance,
                                           offset=self.beta,
                                           scale=self.gamma,
                                           variance_epsilon=self.epsilon)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

net = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=6,kernel_size=5),
    BatchNormalization(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16,kernel_size=5),
    BatchNormalization(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120),
    BatchNormalization(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.Dense(84),
    BatchNormalization(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.Dense(10,activation='sigmoid')]
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

net.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
history = net.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = net.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])


net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Conv2D(filters=6,kernel_size=5))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
net.add(tf.keras.layers.Conv2D(filters=16,kernel_size=5))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(120))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.Dense(84))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.Dense(10,activation='sigmoid'))

# x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
# x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

net.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
history = net.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = net.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

