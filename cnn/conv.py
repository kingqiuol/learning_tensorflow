import tensorflow as tf

def corr2d(X, K):
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j].assign(tf.cast(tf.reduce_sum(X[i:i+h, j:j+w] * K), dtype=tf.float32))
    return Y

X = tf.constant([[0,1,2], [3,4,5], [6,7,8]])
K = tf.constant([[0,1], [2,3]])
print(corr2d(X, K))


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, kernel_size):
        self.w = self.add_weight(name='w',
                                 shape=kernel_size,
                                 initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b',
                                 shape=(1,),
                                 initializer=tf.random_normal_initializer())

    def call(self, inputs):
        return corr2d(inputs, self.w) + self.b

X = tf.Variable(tf.ones((6,8)))
X[:, 2:6].assign(tf.zeros(X[:,2:6].shape))
K = tf.constant([[1,-1]], dtype = tf.float32)
Y = corr2d(X, K)
X = tf.reshape(X, (1,6,8,1))
Y = tf.reshape(Y, (1,6,7,1))

conv2d = tf.keras.layers.Conv2D(1, (1,2))
#input_shape = (samples, rows, cols, channels)
# Y = conv2d(X)

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        dl = g.gradient(l, conv2d.weights[0])
        lr = 3e-2
        update = tf.multiply(lr, dl)
        updated_weights = conv2d.get_weights()
        updated_weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(updated_weights)

        if (i + 1) % 2 == 0:
            print('batch %d, loss %.3f' % (i + 1, tf.reduce_sum(l)))