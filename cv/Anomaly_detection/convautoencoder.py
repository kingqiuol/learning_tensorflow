import tensorflow as tf
import numpy as np


class ConvAutoencoder:
    @staticmethod
    def build(width,height,depth=None,filters=(32,64),latentDim=16):
        '''
        构建自动编码器模型
        :param width: 图像的宽度
        :param height: 图像的高度
        :param depth: 图像的深度
        :param filters: 卷积核的尺寸
        :param latentDim: 全连接的维数
        :return:
        '''
        #定义解码器
        inputs=tf.keras.layers.Input(shape=(height,width,depth))

        x=inputs
        for filter in filters:
            x=tf.keras.layers.Conv2D(filter,kernel_size=(3,3),strides=2,padding='same')(x)
            x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x=tf.keras.layers.BatchNormalization(axis=-1)(x)

        volumeSize=tf.keras.backend.int_shape(x)
        x=tf.keras.layers.Flatten()(x)
        latent=tf.keras.layers.Dense(latentDim)(x)

        encoder=tf.keras.Model(inputs=inputs,outputs=latent,name='encoder')

        #定义编码器
        latentinputs=tf.keras.layers.Input(shape=(latentDim,))

        x=tf.keras.layers.Dense(np.prod(volumeSize[1:]))(latentinputs)
        x=tf.keras.layers.Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)

        for filter in filters[::-1]:
            x=tf.keras.layers.Conv2DTranspose(filter,kernel_size=(3,3),strides=2,padding='same')(x)
            x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x=tf.keras.layers.BatchNormalization(axis=-1)(x)

        x=tf.keras.layers.Conv2DTranspose(depth,(3,3),padding='same')(x)
        outputs=tf.keras.layers.Activation('sigmoid')(x)

        decoder=tf.keras.Model(latentinputs,outputs,name='decoder')

        autoencoder=tf.keras.Model(inputs,decoder(encoder(inputs)),name='autoencoder')

        return (encoder,decoder,autoencoder)

if __name__=="__main__":
    encoder,decoder,autoencoder=ConvAutoencoder.build(28,28,1)
    encoder.summary()
    decoder.summary()
    autoencoder.summary()



