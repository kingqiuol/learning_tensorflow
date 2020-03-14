import tensorflow as tf
import numpy as np
import preprocessing

def load_data(positive_data_file,negative_data_file):
    '''加载数据'''
    file_list = [positive_data_file, negative_data_file]

    def get_line(file_list):
        for file in file_list:
            with open(file,'r',encoding='utf-8') as fp:
                for line in fp:
                    yield line

    x_text=get_line(file_list)#获取文本行
    lenlist=[len(x.split(" ")) for x in x_text]#获取每一行长度
    max_len=max(lenlist)#计算最大长度

    #实例化VocabularyProcessor类
    vocab_processor = preprocessing.VocabularyProcessor(max_len, 5)
    x_text = get_line(file_list)

    # 并将文本转化为向量
    doc=list(vocab_processor.transform(x_text))
    # 生成字典
    dict=list(vocab_processor.reverse([list(range(0, len(vocab_processor.vocabulary_)))]))

    return doc,vocab_processor,dict,max_len


def dataset(positive_data_file,negative_data_file,batch_size=256):
    '''生成数据集'''
    #读取数据集
    doc,vocab_processor,dict,max_len=load_data(positive_data_file,negative_data_file)
    # 生成one_hot标签
    labels=[]
    lenlist=len(doc)
    for i in range(lenlist):
        if i< lenlist/2:
            labels.append(np.array([1,0]))
        else:
            labels.append(np.array([0,1]))

    data=tf.compat.v1.data.Dataset.from_tensor_slices((doc,labels))
    data=data.shuffle(lenlist)
    data=data.batch(batch_size,drop_remainder=True)
    data=data.prefetch(tf.data.experimental.AUTOTUNE)
    return data,vocab_processor,max_len #返回数据集、字典、最大长度

if __name__=="__main__":
    positive_data_file = "./rt-polaritydata/rt-polarity.pos"
    negative_data_file = "./rt-polaritydata/rt-polarity.neg"
    data,vocab_processor,max_len=dataset(positive_data_file,negative_data_file)
    print("字典：",list(vocab_processor.reverse([list(range(0, len(vocab_processor.vocabulary_)))])))


    # iterator = data.make_one_shot_iterator()  # 从到到尾读一次
    # one_element = iterator.get_next()  # 从iterator里取出一个元素
    #
    # with tf.compat.v1.Session() as sess:
    #     for i in range(10):
    #         print(one_element)







