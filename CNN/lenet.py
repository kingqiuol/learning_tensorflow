import tensorflow as tf

#加载数据
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#搭建模型
def LeNet(num_classes):
    inputs=tf.keras.layers.Input(shape=[28,28,1])
    x=tf.keras.layers.Conv2D(6,5,activation="sigmoid",name="conv1")(inputs)
    x=tf.keras.layers.MaxPooling2D(2,2,name="pool1")(x)

    x = tf.keras.layers.Conv2D(16, 5, activation="sigmoid", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D(2, 2, name="pool2")(x)

    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(120,activation="sigmoid",name="fc3")(x)
    x=tf.keras.layers.Dense(84,activation="sigmoid",name="fc4")(x)
    x=tf.keras.layers.Dense(num_classes,activation="sigmoid",name="output")(x)

    model=tf.keras.Model(inputs=inputs,outputs=x)

    return model

lenet=LeNet(10)
lenet.summary()

train_images = tf.reshape(train_images, (train_images.shape[0],train_images.shape[1],train_images.shape[2], 1))
test_images = tf.reshape(test_images, (test_images.shape[0],test_images.shape[1],test_images.shape[2], 1))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.9, momentum=0.0, nesterov=False)

lenet.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
lenet.fit(train_images, train_labels, epochs=5, validation_split=0.1)

