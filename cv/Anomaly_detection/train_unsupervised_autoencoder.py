import matplotlib
matplotlib.use("Agg")

from convautoencoder import ConvAutoencoder

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2

from sklearn.model_selection import train_test_split




def build_unsupervised_dataset(data,labels,validLabel=1,
	anomalyLabel=3, contam=0.01, seed=42):
    '''制作数据集'''

    # 抓取提供的类标签的所有 * 真正 * 索引特定的标签，
    # 然后获取图像的索引个标签将成为我们的“异常”
    validIdxs=np.where(labels==validLabel)[0]
    anomalyLabelIdx=np.where(labels==anomalyLabel)[0]

    #随机打乱数据
    random.shuffle(validIdxs)
    random.shuffle(anomalyLabelIdx)

    #计算并设置异常数据的个数
    i=int(len(validIdxs)*contam)
    anomalyLabelIdx=anomalyLabelIdx[:i]

    #提取正常数据和异常数据
    validImages=data[validIdxs]
    anomalyImages=data[anomalyLabelIdx]

    #打包数据并进行数据打乱
    images=np.vstack([validImages,anomalyImages])

    return images

def visualize_predictions(decoded,gt,samples=10):
    '''可视化预测结果'''

    outputs=None

    for i in range(samples):
        original=(gt[i]*255).astype('uint8')
        recon=(decoded[i]*255).astype('uint8')

        output=np.hstack([original,recon])

        if outputs is None:
            outputs=output
        else:
            outputs=np.vstack([outputs,output])

    return outputs

arg=argparse.ArgumentParser()

arg.add_argument("-d", "--dataset", type=str, required=True,
	help="path to output dataset file")
arg.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained autoencoder")
arg.add_argument("-v", "--vis", type=str, default="recon_vis.png",
	help="path to output reconstruction visualization file")
arg.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output plot file")


tf.keras.backend.clear_session()

epochs=20
lr=1e-3
batch_size=32

# 加载MNIST数据集
print("[INFO] loading MNIST dataset...")
((trainX,trainY),(testX,testY))=tf.keras.datasets.mnist.load_data()


# 建立少量的无监督图像数据集,污染（即异常）添加到其中
print("[INFO] creating unsupervised dataset...")
images = build_unsupervised_dataset(trainX, trainY, validLabel=1,
	anomalyLabel=3, contam=0.01)

# 构建训练和测试分组
(trainX,testX)=train_test_split(images,test_size=0.2,random_state=42)

trainX = trainX.reshape(-1, 28, 28, 1)
trainX = trainX.astype('float32')
trainX/=255

testX = testX.reshape(-1, 28, 28, 1)
testX = testX.astype('float32')
testX/=255


#搭建模型
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28,1)

#搭建优化器
opt=tf.keras.optimizers.Adam(lr=lr,decay=lr/epochs)
autoencoder.compile(loss='mse',optimizer=opt,metrics=['acc'])

#训练
H=autoencoder.fit(trainX,trainX,validation_data=(testX,testX),epochs=epochs,batch_size=batch_size)

print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite("recon_vis.png", vis)

N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot.png")
# serialize the image data to disk
print("[INFO] saving image data...")
f = open("./mages.pickle", "wb")
f.write(pickle.dumps(images))
f.close()
# serialize the autoencoder model to disk
print("[INFO] saving autoencoder...")
autoencoder.save("autoencoder.h5")
