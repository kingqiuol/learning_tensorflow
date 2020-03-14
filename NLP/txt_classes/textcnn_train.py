import tensorflow as tf
from textcnn import TextCNN
from load_data import dataset



def train():
    # 指定样本文件
    positive_data_file = "./rt-polaritydata/rt-polarity.pos"
    negative_data_file = "./rt-polaritydata/rt-polarity.neg"
    # 设置训练参数
    num_steps = 50  # 定义训练次数
    SaveFileName = "text_cnn_model"  # 定义保存模型文件夹名称
    # 设置模型参数
    num_classes = 2  # 设置模型分类
    l2_reg_lambda = 0.1  # 定义正则化系数
    filter_sizes = "3,4,5"  # 定义多通道卷积核
    num_filters = 64  # 定义每通道的输出个数

    # 加载数据集
    data,vocab_processor,max_len=dataset(positive_data_file,negative_data_file)
    #搭建模型
    text_cnn=TextCNN(seq_length=max_len,
                     num_classes=num_classes,
                     vocab_size=len(vocab_processor.vocabulary_),
                     embeding_size=128,filter_sizes=list(map(int,filter_sizes.split(','))),
                     num_filters=num_filters)


    def l2_loss(y_true, y_pred):
        l2_loss=tf.constant(0.0)
        for tf_var in text_cnn.trainable_weights:
            if tf_var.name == "fully_connecred":
                l2_loss+=tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true)
        return loss+l2_reg_lambda*l2_loss


    text_cnn.compile(loss=l2_loss,
                     optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                     metrics=['acc'])
    text_cnn.fit(data,epochs=num_steps)

    text_cnn.save("textcnn.h5")

train()


# 2020-03-14 12:06:25.063619: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Out of range: End of sequence
# 	 [[{{node IteratorGetNext}}]]