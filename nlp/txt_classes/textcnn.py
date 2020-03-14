import tensorflow as tf

def TextCNN(seq_length,num_classes,vocab_size,embeding_size,filter_sizes,num_filters,dropout_keep_prob=0.8):
    '''构建TextCNN模型'''
    input_x=tf.keras.layers.Input([seq_length])

    # 词嵌入层
    out=tf.keras.layers.Embedding(vocab_size,embeding_size)(input_x)
    out=tf.expand_dims(out,-1)

    # 定义多通道卷积和最大池化层
    pooled_output=[]
    for i ,filter_size in enumerate(filter_sizes):
        conv=tf.keras.layers.Conv2D(num_filters,
                                    kernel_size=[filter_size,embeding_size],
                                    strides=1,padding='valid',
                                    activation='relu',
                                    name="conv%s"%filter_size)(out)

        pooled=tf.keras.layers.MaxPooling2D(pool_size=[seq_length-filter_size+1,1],
                                            padding='valid',
                                            name="pool%s"%filter_size)(conv)

        pooled_output.append(pooled)#将各通道的结果合并起来

    #展开特征，并添加dropout方法
    num_filters_total=num_filters*len(filter_sizes)
    h_pool=tf.concat(pooled_output,3)
    h_pool_flat=tf.reshape(h_pool,[-1,num_filters_total])

    out=tf.keras.layers.Dropout(rate=dropout_keep_prob,name="dropout")(h_pool_flat)

    out=tf.keras.layers.Dense(num_classes,activation='softmax',name="fully_connecred")(out)

    model=tf.keras.Model(inputs=input_x,outputs=out,name="textcnn")

    return model

if __name__=="__main__":
    # 设置模型参数
    num_classes = 2  # 设置模型分类
    dropout_keep_prob = 0.8  # 定义dropout系数
    l2_reg_lambda = 0.1  # 定义正则化系数
    filter_sizes = "3,4,5"  # 定义多通道卷积核
    num_filters = 64  # 定义每通道的输出个数

    textcnn=TextCNN(100,num_classes,1000,128,list(map(int, filter_sizes.split(","))),num_filters,dropout_keep_prob)
    textcnn.summary()




