import tensorflow as tf
import numpy as np


class DataLoader():
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(self.train_images.astype(np.float32) / 255.0, axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32) / 255.0, axis=-1)
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        # need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.train_images[index], 224, 224, )
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        # need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.test_images[index], 224, 224, )
        return resized_images.numpy(), self.test_labels[index]


batch_size = 128
dataLoader = DataLoader()
x_batch, y_batch = dataLoader.get_batch_train(batch_size)
print("x_batch shape:", x_batch.shape, "y_batch shape:", y_batch.shape)

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                       padding='same', activation='relu'))

    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk

def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs,num_channels))
    net.add(tf.keras.models.Sequential([tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(4096,activation='relu'),
             tf.keras.layers.Dropout(0.5),
             tf.keras.layers.Dense(4096,activation='relu'),
             tf.keras.layers.Dropout(0.5),
             tf.keras.layers.Dense(10,activation='sigmoid')]))
    return net


ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)


def train_vgg():
    #     net.load_weights("5.7_vgg_weights.h5")
    epoch = 5
    num_iter = dataLoader.num_train // batch_size
    for e in range(epoch):
        for n in range(num_iter):
            x_batch, y_batch = dataLoader.get_batch_train(batch_size)
            net.fit(x_batch, y_batch)
            if n % 20 == 0:
                net.save_weights("5.7_vgg_weights.h5")


optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0, nesterov=False)

net.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

x_batch, y_batch = dataLoader.get_batch_train(batch_size)
net.fit(x_batch, y_batch)
# train_vgg()