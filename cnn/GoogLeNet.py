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

class Inception(tf.keras.layers.Layer):
    def __init__(self,c1,c2,c3,c4):
        super(Inception,self).__init__()
        #1
        self.p1=tf.keras.layers.Conv2D(c1,kernel_size=1,activation="relu",padding='same')

        #2
        self.p2_1=tf.keras.layers.Conv2D(c2[0],kernel_size=1,padding='same',activation='relu')
        self.p2_2=tf.keras.layers.Conv2D(c2[1],kernel_size=3,padding='same',activation='relu')

        #3
        self.p3_1=tf.keras.layers.Conv2D(c3[0],kernel_size=1,padding='same',activation='relu')
        self.p3_2=tf.keras.layers.Conv2D(c3[1],kernel_size=5,padding='same',activation='relu')

        #4
        self.p4_1 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same', strides=1)
        self.p4_2 = tf.keras.layers.Conv2D(c4, kernel_size=1, padding='same', activation='relu')

    def call(self, inputs):
        p1=self.p1(inputs)

        p2=self.p2_1(inputs)
        p2=self.p2_2(p2)

        p3=self.p3_1(inputs)
        p3=self.p3_2(p3)

        p4=self.p4_1(inputs)
        p4=self.p4_2(p4)

        return tf.concat([p1,p2,p3,p4],axis=-1)

def GoogLeNet(num_classes):
    inputs=tf.keras.layers.Input(shape=[224,224,1])

    #1
    x=tf.keras.layers.Conv2D(64,7,2,padding='same',activation='relu')(inputs)
    x=tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same')(x)

    #2
    x=tf.keras.layers.Conv2D(64,1,padding='same',activation='relu')(x)
    x=tf.keras.layers.Conv2D(192,3,padding='same',activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=3,padding='same')(x)

    #3
    x=Inception(64,(96,128),(16,32),32)(x)
    x=Inception(128, (128, 192), (32, 96), 64)(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same')(x)

    #4
    x=Inception(192, (96, 208), (16, 48), 64)(x)
    x=Inception(160, (112, 224), (24, 64), 64)(x)
    x=Inception(128, (128, 256), (24, 64), 64)(x)
    x=Inception(112, (144, 288), (32, 64), 64)(x)
    x=Inception(256, (160, 320), (32, 128), 128)(x)
    x=tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    #5
    x=Inception(256, (160, 320), (32, 128), 128)(x)
    x=Inception(384, (192, 384), (48, 128), 128)(x)
    x=tf.keras.layers.GlobalAvgPool2D()(x)

    #6
    x=tf.keras.layers.Dense(num_classes,activation='sigmoid')(x)

    model=tf.keras.Model(inputs=inputs,outputs=x)

    return model

net=GoogLeNet(10)
net.summary()


def train_googlenet():
    net.load_weights("5.9_googlenet_weights.h5")
    epoch = 5
    num_iter = dataLoader.num_train // batch_size
    for e in range(epoch):
        for n in range(num_iter):
            x_batch, y_batch = dataLoader.get_batch_train(batch_size)
            net.fit(x_batch, y_batch)
            if n % 20 == 0:
                net.save_weights("5.9_googlenet_weights.h5")


# optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0, nesterov=False)
optimizer = tf.keras.optimizers.Adam(lr=1e-7)

net.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

x_batch, y_batch = dataLoader.get_batch_train(batch_size)
net.fit(x_batch, y_batch)
train_googlenet()
net.load_weights("5.9_googlenet_weights.h5")

x_test, y_test = dataLoader.get_batch_test(2000)
net.evaluate(x_test, y_test, verbose=2)


