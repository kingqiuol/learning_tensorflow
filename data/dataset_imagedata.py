import os
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

def load_sample(sample_dir,shuffleflag = True):
    '''递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名'''
    print ('loading sample  dataset..')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#递归遍历文件夹
        for filename in filenames:                            #遍历所有文件名
            #print(dirnames)
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)               #添加文件名
            labelsnames.append( dirpath.split('\\')[-1] )#添加文件名对应的标签

    lab= list(sorted(set(labelsnames)))  #生成标签名称列表
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #生成字典

    labels = [labdict[i] for i in labelsnames]
    if shuffleflag == True:
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    else:
        return (np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)

def _norm_image(image,size,ch=1,flattenflag = False):    #定义函数，实现归一化，并且拍平
    image_decoded = image/255.0
    if flattenflag==True:
        image_decoded = tf.reshape(image_decoded, [size[0]*size[1]*ch])
    return image_decoded

def dataset(directory,size,batchsize,random_rotated=False):#定义函数，创建数据集
    """ parse  dataset."""
    (filenames,labels),_ =load_sample(directory,shuffleflag=False) #载入文件名称与标签
    def _parseone(filename, label):                         #解析一个图片文件
        """ Reading and handle  image"""
        image_string = tf.read_file(filename)         #读取整个文件
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape([None, None, None])    # 必须有这句，不然下面会转化失败
        image_decoded = tf.image.resize(image_decoded, size)  #变化尺寸
        image_decoded = _norm_image(image_decoded,size)#归一化
        image_decoded = tf.cast(image_decoded,dtype=tf.float32)
        label = tf.cast(  tf.reshape(label, []) ,dtype=tf.int32  )#将label 转为张量
        return image_decoded, label

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))#生成Dataset对象
    dataset = dataset.map(_parseone)   #有图片内容的数据集

    dataset = dataset.batch(batchsize) #批次划分数据集

    return dataset

def showresult(subplot,title,thisimg):          #显示单个图片
    p =plt.subplot(subplot)
    p.axis('off')
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):   #显示
    plt.figure(figsize=(20,10))     #定义显示图片的宽、高
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()

def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#生成一个迭代器
    one_element = iterator.get_next()					#从iterator里取出一个元素
    return one_element

sample_dir=r"./hymenoptera_data/train"
size = [96,96]
batchsize = 10
tdataset = dataset(sample_dir,size,batchsize)
tdataset2 = dataset(sample_dir,size,batchsize,True)
print(tdataset.output_types)  #打印数据集的输出信息
print(tdataset.output_shapes)

one_element1 = getone(tdataset)				#从tdataset里取出一个元素
one_element2 = getone(tdataset2)				#从tdataset2里取出一个元素


with tf.Session() as sess:	# 建立会话（session）
    sess.run(tf.global_variables_initializer())  #初始化

    try:
        for step in np.arange(1):
            value = sess.run(one_element1)
            value2 = sess.run(one_element2)

            showimg(step,value[1],np.asarray( value[0]*255,np.uint8),10)       #显示图片
            showimg(step,value2[1],np.asarray( value2[0]*255,np.uint8),10)       #显示图片


    except tf.errors.OutOfRangeError:           #捕获异常
        print("Done!!!")
