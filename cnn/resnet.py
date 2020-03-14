import tensorflow as tf
from tensorflow.keras import layers,activations
class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=strides)
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3,padding='same')
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels,
                                       kernel_size=1,
                                       strides=strides)
        else:
            self.conv3 = None
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, X):
        Y = activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return activations.relu(Y + X)

blk = Residual(3)
#tensorflow input shpe     (n_images, x_shape, y_shape, channels).
#mxnet.gluon.nn.conv_layers    (batch_size, in_channels, height, width)
X = tf.random.uniform((4, 6, 6 , 3))
print(blk(X).shape)#TensorShape([4, 6, 6, 3])

blk = Residual(6, use_1x1conv=True, strides=2)
print(blk(X).shape)
#TensorShape([4, 3, 3, 6])

net = tf.keras.models.Sequential(
    [layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    layers.BatchNormalization(), layers.Activation('relu'),
    layers.MaxPool2D(pool_size=3, strides=2, padding='same')])


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.listLayers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.listLayers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.listLayers.layers:
            X = layer(X)
        return X


class ResNet(tf.keras.Model):
    def __init__(self, num_blocks, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.conv = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.mp = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.resnet_block1 = ResnetBlock(64, num_blocks[0], first_block=True)
        self.resnet_block2 = ResnetBlock(128, num_blocks[1])
        self.resnet_block3 = ResnetBlock(256, num_blocks[2])
        self.resnet_block4 = ResnetBlock(512, num_blocks[3])
        self.gap = layers.GlobalAvgPool2D()
        self.fc = layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.resnet_block4(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


mynet = ResNet([2, 2, 2, 2])

X = tf.random.uniform(shape=(1,  224, 224 , 1))
for layer in mynet.layers:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

mynet.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

history = mynet.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = mynet.evaluate(x_test, y_test, verbose=2)

