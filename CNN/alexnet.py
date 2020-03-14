import tensorflow as tf
import numpy as np
#加载数据
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

def AlexNet(num_classes):
    inputs=tf.keras.layers.Input(shape=[224,224,1])
    x=tf.keras.layers.Conv2D(96,11,4,activation="relu",name='conv1')(inputs)
    x=tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,name='pool1')(x)

    x=tf.keras.layers.Conv2D(256,5,padding='same',activation='relu',name="conv_2")(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,name='pool2')(x)

    x=tf.keras.layers.Conv2D(filters=384,kernel_size=3,padding='same',activation='relu',name="conv3")(x)
    x=tf.keras.layers.Conv2D(filters=384,kernel_size=3,padding='same',activation='relu',name="conv4")(x)
    x=tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu',name="conv5")(x)
    x=tf.keras.layers.MaxPool2D(pool_size=3, strides=2,name="pool5")(x)

    x=tf.keras.layers.Flatten()(x)

    x=tf.keras.layers.Dense(4096,activation='relu',name="fc6")(x)
    x=tf.keras.layers.Dropout(0.5,name="dropout6")(x)

    x=tf.keras.layers.Dense(4096,activation='relu',name="fc7")(x)
    x=tf.keras.layers.Dropout(0.5,name="droupout7")(x)

    x=tf.keras.layers.Dense(num_classes,activation='sigmoid',name="fc8")(x)
    model=tf.keras.Model(inputs=inputs,outputs=x)

    return model

alexnet=AlexNet(10)
alexnet.summary()


def train_alexnet():
    epoch = 5
    num_iter = dataLoader.num_train // batch_size
    for e in range(epoch):
        for n in range(num_iter):
            x_batch, y_batch = dataLoader.get_batch_train(batch_size)
            alexnet.fit(x_batch, y_batch)
            if n % 20 == 0:
                alexnet.save_weights("5.6_alexnet_weights.h5")


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

alexnet.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# x_batch, y_batch = dataLoader.get_batch_train(batch_size)
# alexnet.fit(x_batch, y_batch)
train_alexnet()

alexnet.load_weights("5.6_alexnet_weights.h5")

x_test, y_test = dataLoader.get_batch_test(2000)
alexnet.evaluate(x_test, y_test, verbose=2)
