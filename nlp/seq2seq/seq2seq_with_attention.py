import tensorflow as tf


# 编码器模型
class DNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DNN_Encoder, self).__init__()
        # keras的全连接支持多维输入。仅对最后一维进行处理
        self.fc = tf.keras.layers.Dense(embedding_dim)  # (batch_size, 49, embedding_dim)

    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs)
        x = tf.keras.layers.Activation('relu')(x)

        return x


def gru(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')


# 注意力模型
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features,  # features形状(batch_size, 49, embedding_dim)
             hidden):  # hidden(batch_size, hidden_size)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # (batch_size, 1, hidden_size)

        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))  # (batch_size, 49, hidden_size)

        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch_size, 49, 1)

        context_vector = attention_weights * features  # (batch_size, 49, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size,  hidden_size)

        return context_vector, attention_weights


# 解码器模型
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.units)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, inputs, features, hidden, training=None, mask=None):
        # 返回注意力特征向量和注意力权重
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(inputs)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  # (batch_size, 1, embedding_dim + hidden_size)
        output, state = self.gru(x)  # 使用循环网络gru进行处理

        x = self.fc1(output)  # (batch_size, max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))  # (batch_size * max_length, hidden_size)

        x = self.fc2(x)  # (batch_size * max_length, vocab)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
