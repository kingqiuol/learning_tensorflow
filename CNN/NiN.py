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

def _nin_block(num_channels,kernel_size,strides,padding):
    blk=tf.keras.models.Sequential()
    blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size,strides,padding,activation="relu"))
    blk.add(tf.keras.layers.Conv2D(num_channels,1,activation="relu"))
    blk.add(tf.keras.layers.Conv2D(num_channels,1,activation='relu'))
    return blk

def NiN(num_classes):
    inputs=tf.keras.layers.Input(shape=[224,224,1])
    x=_nin_block(96,11,4,'valid')(inputs)
    x=tf.keras.layers.MaxPooling2D(3,2)(x)
    x=_nin_block(256,5,1,padding='same')(x)
    x=tf.keras.layers.MaxPooling2D(3,2)(x)
    x=_nin_block(384,3,1,padding='same')(x)
    x=tf.keras.layers.MaxPooling2D(3,2)(x)
    x=tf.keras.layers.Dropout(0.5)(x)
    x=_nin_block(num_classes,3,1,padding='same')(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Flatten()(x)

    model=tf.keras.Model(inputs=inputs,outputs=x)
    return model

nin=NiN(10)
nin.summary()


def train_nin():
    nin.load_weights("5.8_nin_weights.h5")
    epoch = 5
    num_iter = dataLoader.num_train // batch_size
    for e in range(epoch):
        for n in range(num_iter):
            x_batch, y_batch = dataLoader.get_batch_train(batch_size)
            nin.fit(x_batch, y_batch)
            if n % 20 == 0:
                nin.save_weights("5.8_nin_weights.h5")


# optimizer = tf.keras.optimizers.SGD(learning_rate=0.06, momentum=0.3, nesterov=False)
optimizer = tf.keras.optimizers.Adam(lr=1e-7)
nin.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

x_batch, y_batch = dataLoader.get_batch_train(batch_size)
nin.fit(x_batch, y_batch)
# train_nin()