import numpy as np
import os
import shutil
import json
import tensorflow as tf
from pycocotools.coco import COCO
from tensorflow.python.keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split

# ResNet50预训练权重
RESNET50_WEIGHTS = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# 选取选练样本的数量（注意为了方便训练演示选取部分数据集）
NUM_EXAMPLE = 300
# 限定字典的最大长度
MAX_VOCAB_SIZE = 5000


def make_numpy_feature(numpyPATH, img_filename, PATH, weights=RESNET50_WEIGHTS):
    '''
    将图片转化为特征数据
    :param numpyPATH:存储提取好特征后的存储文件夹
    :param img_filename:图片文件列表
    :param PATH: 图片所在的文件夹
    :param weights: ResNet50模型的权重
    :return:
    '''
    if os.path.exists(numpyPATH):  # 去除已有文件夹
        shutil.rmtree(numpyPATH, ignore_errors=True)

    os.mkdir(numpyPATH)  # 新建文件夹

    size = [224, 224]  # 设置输出尺寸
    batch_size = 10  # 批量大小

    def load_image(image_path):
        '''输入图片的预处理'''
        img = tf.compat.v1.read_file(PATH + image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)
        # 使用Reset模型的统一预处理
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, image_path

    # 创建ResNet模型
    image_model = ResNet50(weights=weights, include_top=False)
    new_input = image_model.input
    # 获取ResNet导数第二层（池化前的卷积结果）
    hidden_layer = image_model.layers[-2].output
    image_feature_extract_model = tf.keras.Model(new_input, hidden_layer)
    image_feature_extract_model.summary()

    # 对文件目录去重
    encode_train = sorted(set(img_filename))

    # 图片数据集
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(load_image).batch(batch_size)

    for img, path in image_dataset:
        batch_feature = image_feature_extract_model(img)
        print(batch_feature.shape)
        batch_feature = tf.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))
        print(batch_feature.shape)
        for bf, p in zip(batch_feature, path):
            path_of_feature = p.numpy().decode('utf-8')
            np.save(numpyPATH + path_of_feature, bf.numpy())


def text_preprocessing(train_caption, max_vocab_size):
    '''
    文本标签预处理：（1）文本过滤；（2）建立字典；（3）向量化文本以及文本对齐
    :param train_caption: 文本标签数据集
    :param max_vocab_size: 限制最大字典的大小
    :return:
    '''
    # 文本过滤，去除无效字符
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab_size,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_caption)

    # 建立字典，构造正反向字典
    tokenizer.word_index = {key: value for key, value in tokenizer.word_index.items() if value <= max_vocab_size}
    # 向字典中加入<unk>字符
    tokenizer.word_index[tokenizer.oov_token] = max_vocab_size + 1
    # 向字典中加入<pad>字符
    tokenizer.word_index['<pad>'] = 0

    index_word = {value: key for key, value in tokenizer.word_index.items()}

    # 向量化文本和对齐操作，将文本按照字典的数字进行项向量化处理，
    # 并按照指定长度进行对齐操作（多余的截调，不足的进行补零）
    train_seqs = tokenizer.texts_to_sequences(train_caption)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length = len(cap_vector[0])  # 标签最大长度

    return cap_vector, max_length, tokenizer.word_index, index_word


def load_data(annotation_file, PATH, numpyPATH,
              num_example=NUM_EXAMPLE, max_vocab_size=MAX_VOCAB_SIZE):
    '''
    对数据集进行预处理并加载数据集
    :param annotation_file: 训练数据的标注文件
    :param PATH: 图片数据集
    :param numpyPATH: 将图片提取特后的存储的位置
    :param num_example: 这里选择其中300个样本数据（注意为了方便训练演示，你也可以训练全部数据集）
    :param max_vocab_size: 限定字典的最大长度
    :return:
    '''
    with open(annotation_file, 'r') as f:  # 加载标注文件
        annotations = json.load(f)

    train_caption = []  # 存储图片对应的标题
    img_filenames = []  # 存储图片的路径

    # 获取全部文件及对应的标注文本
    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        img_id = annot['image_id']

        full_coco_image_path = 'COCO_train2014_' + "%012d.jpg" % (img_id)

        img_filenames.append(full_coco_image_path)
        train_caption.append(caption)

        if len(train_caption) >= num_example:
            break

    # 将图片转化为特征数据，并进行存储
    if os.path.exists(numpyPATH):
        make_numpy_feature(numpyPATH, img_filenames, PATH)

    # 文本数据的预处理
    cap_vector, max_length, word_index, index_word = text_preprocessing(train_caption, max_vocab_size)

    # 将数据拆分为训练集和测试集
    img_name_train, img_name_val, cap_train, cap_val = \
        train_test_split(img_filenames, cap_vector, test_size=0.2, random_state=0)

    return img_name_train, cap_train, img_name_val, cap_val, max_length, word_index, index_word


def dataset(annotation_file, PATH, numpyPATH, batch_size):
    '''
    创建数据集
    :param instances_file: 训练数据
    :param annotation_file: 训练数据的标注文件
    :param PATH: 图片数据集
    :param numpyPATH: 将图片提取特后的存储的位置
    :param batch_size: 数据集的batch size
    :return:
    '''
    img_name_train, cap_train, img_name_val, cap_val, max_length, word_index, index_word = \
        load_data(annotation_file, PATH, numpyPATH)

    def map_func(img_name, cap):
        #         print("===========================================",numpyPATH+str(img_name.numpy()).split("\'")[1]+'.npy')
        img_tensor = np.load(numpyPATH + str(img_name.numpy()).split("\'")[1] + '.npy')
        return img_tensor, cap

    train_dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    train_dataset = train_dataset.map(lambda item1, item2: tf.py_function(
        map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)

    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(1)

    return train_dataset, img_name_val, cap_val, max_length, word_index, index_word


if __name__ == "__main__":
    #     image_model=ResNet50(weights=RESNET50_WEIGHTS
    #                          ,include_top=False)
    #     new_input=image_model.input
    #     #获取ResNet导数第二层（池化前的卷积结果）
    #     hidden_layer=image_model.layers[-2].output
    #     image_feature_extract_model=tf.keras.Model(new_input,hidden_layer)
    #     image_feature_extract_model.summary()
    # 加载数据
    batch_size = 20
    annotation_file = r'annotations/captions_train2014.json'
    PATH = r"train2014/"
    numpyPATH = './numpyfeature/'
    dataset, img_name_val, cap_val, max_length, word_index, index_word = dataset(annotation_file, PATH, numpyPATH,
                                                                                 batch_size)


