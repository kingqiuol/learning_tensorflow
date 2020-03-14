import collections
import math
import random
import sys
import time
import os
import numpy as np
import tensorflow as tf

###########################################################################
'''
一、PTB 数据集
Word2Vec 能从语料中学到如何将离散的词映射为连续空间中的向量，并保留其语义上的相似关系。

我们使用经典的 PTB 语料库进行训练。PTB (Penn Tree Bank) 是一个常用的小型语料库，它采
样自《华尔街日报》的文章，包括训练集、验证集和测试集。我们将在PTB训练集上训练词嵌入模型。
'''
###########################################################################

#1、载入数据集
with open('./ptb.train.txt', 'r') as f:
    lines = f.readlines() # 该数据集中句子以换行符为分割
    raw_dataset = [st.split() for st in lines] # st是sentence的缩写，单词以空格为分割
print('# sentences: %d' % len(raw_dataset))
#
# # 对于数据集的前3个句子，打印每个句子的词数和前5个词
# # 句尾符为 '' ，生僻词全用 '' 表示，数字则被替换成了 'N'
# for st in raw_dataset[:3]:
#     print('# tokens:', len(st), st[:5])

#2、建立词语索引
counter = collections.Counter([tk for st in raw_dataset for tk in st]) # tk是token的缩写
counter = dict(filter(lambda x: x[1] >= 5, counter.items())) # 只保留在数据集中至少出现5次的词

idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset] # raw_dataset中的单词在这一步被转换为对应的idx
num_tokens = sum([len(st) for st in dataset])
print('# tokens: %d' % num_tokens)

#3、二次采样
'''
文本数据中一般会出现一些高频词，如英文中的“the”“a”和“in”。通常来说，在一个背景窗口中，
一个词（如“chip”）和较低频词（如“microprocessor”）同时出现比和较高频词（如“the”）
同时出现对训练词嵌入模型更有益。因此，训练词嵌入模型时可以对词进行二次采样。
'''
def discard(idx):
    '''
    @params:
        idx: 单词的下标
    @return: True/False 表示是否丢弃该单词
    '''
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print('# tokens: %d' % sum([len(st) for st in subsampled_dataset]))

def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

# print(compare_counts('the'))
# print(compare_counts('join'))

#4、提取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    '''
    @params:
        dataset: 数据集为句子的集合，每个句子则为单词的集合，此时单词已经被转换为相应数字下标
        max_window_size: 背景词的词窗大小的最大值
    @return:
        centers: 中心词的集合
        contexts: 背景词窗的集合，与中心词对应，每个背景词窗则为背景词的集合
    '''
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size) # 随机选取背景词窗大小
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

# tiny_dataset = [list(range(7)), list(range(7, 10))]
# print('dataset', tiny_dataset)
# for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
#     print('center', center, 'has contexts', context)

#5、负采样近似
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

#6、读取数据
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        center=center.numpy().tolist()
        context=context.numpy().tolist()
        negative=negative.numpy().tolist()
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return tf.data.Dataset.from_tensor_slices((tf.reshape(tf.convert_to_tensor(centers),shape=(-1, 1)), tf.convert_to_tensor(contexts_negatives),
            tf.convert_to_tensor(masks), tf.convert_to_tensor(labels)))

def generator():
    for cent, cont, neg in zip(all_centers,all_contexts,all_negatives):
        yield (cent, cont, neg)

batch_size = 512
dataset=tf.data.Dataset.from_generator(generator=generator,output_types=(tf.int32,tf.int32, tf.int32))
dataset = dataset.apply(batchify).shuffle(len(all_centers)).batch(batch_size)

for batch in dataset:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break

###########################################################################
'''
二、Skip-Gram 跳字模型
通过使用嵌入层和小批量乘法来实现跳字模型。
'''
###########################################################################
embed = tf.keras.layers.Embedding(input_dim=20, output_dim=4)
embed.build(input_shape=(1,20))
print(embed.get_weights())

#1、跳字模型前向计算
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = tf.matmul(v, tf.transpose(u,perm=[0,2,1]))
    return pred

###########################################################################
'''
三、训练模型
通过使用嵌入层和小批量乘法来实现跳字模型。
'''
###########################################################################
#1、二元交叉熵损失函数
class SigmoidBinaryCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def __call__(self, inputs, targets, mask=None):
        #tensorflow中使用tf.nn.weighted_cross_entropy_with_logits设置mask并没有起到作用
        #直接与mask按元素相乘回实现当mask为0时不计损失的效果
        inputs=tf.cast(inputs,dtype=tf.float32)
        targets=tf.cast(targets,dtype=tf.float32)
        mask=tf.cast(mask,dtype=tf.float32)
        res=tf.nn.sigmoid_cross_entropy_with_logits(inputs, targets)*mask
        return tf.reduce_mean(res,axis=1)

loss = SigmoidBinaryCrossEntropyLoss()

pred = tf.convert_to_tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]],dtype=tf.float32)
# 标签变量label中的1和0分别代表背景词和噪声词
label = tf.convert_to_tensor([[1, 0, 0, 0], [1, 1, 0, 0]],dtype=tf.float32)
mask = tf.convert_to_tensor([[1, 1, 1, 1], [1, 1, 1, 0]],dtype=tf.float32)  # 掩码变量
print(loss(label, pred, mask) * mask.shape[1] / tf.reduce_sum(mask,axis=1))

def sigmd(x):
    return - math.log(1 / (1 + math.exp(-x)))

print('%.4f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4)) # 注意1-sigmoid(x) = sigmoid(-x)
print('%.4f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))

#2、初始化模型参数
embed_size = 100
net = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
    tf.keras.layers.Embedding(input_dim=len(idx_to_token), output_dim=embed_size)
])
net.get_layer(index=0)

#3、定义训练函数
def train(net, lr, num_epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in dataset:
            center, context_negative, mask, label = [d for d in batch]
            mask = tf.cast(mask, dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                pred = skip_gram(center, context_negative, net.get_layer(index=0), net.get_layer(index=1))
                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                l = (loss(label, tf.reshape(pred, label.shape), mask) *
                     mask.shape[1] / tf.reduce_sum(mask, axis=1))
                l = tf.reduce_mean(l)  # 一个batch的平均loss

            grads = tape.gradient(l, net.variables)
            optimizer.apply_gradients(zip(grads, net.variables))
            l_sum += np.array(l).item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))

train(net, 0.01, 10)


###########################################################################
'''
四、应用词嵌入模型
通过使用嵌入层和小批量乘法来实现跳字模型。
'''
###########################################################################
def get_similar_tokens(query_token, k, embed):
    W = embed.get_weights()
    W = tf.convert_to_tensor(W[0])
    x = W[token_to_idx[query_token]]
    x = tf.reshape(x, shape=[-1, 1])
    # 添加的1e-9是为了数值稳定性
    cos = tf.reshape(tf.matmul(W, x), shape=[-1]) / tf.sqrt(tf.reduce_sum(W * W, axis=1) * tf.reduce_sum(x * x) + 1e-9)
    _, topk = tf.math.top_k(cos, k=k + 1)
    topk = topk.numpy().tolist()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))


get_similar_tokens('chip', 3, net.get_layer(index=0))

