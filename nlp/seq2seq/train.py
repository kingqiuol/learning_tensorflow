import numpy as np
import tensorflow as tf
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

from seq2seq_with_attention import DNN_Encoder, RNN_Decoder
from load_data import dataset

# 加载数据
batch_size = 20
annotation_file = r'annotations/captions_train2014.json'
PATH = r"train2014/"
numpyPATH = './numpyfeature/'
dataset, img_name_val, cap_val, max_length, word_index, index_word = dataset(annotation_file, PATH, numpyPATH,
                                                                             batch_size)

# 模型搭建
embedding_dim = 256
units = 512
vocab_size = len(word_index)  # 字典大小

# 图片特征(47, 2048)
features_shape = 2048
attention_features_shape = 49

# 创建模型对象字典
model_objects = {
    'encoder': DNN_Encoder(embedding_dim),
    'decoder': RNN_Decoder(embedding_dim, units, vocab_size),
    'optimizer': tf.train.AdamOptimizer(),
    'step_counter': tf.train.get_or_create_global_step(),
}

checkpoint_prefix = os.path.join("mytfemodel/", 'ckpt')
checkpoint = tf.train.Checkpoint(**model_objects)
latest_cpkt = tf.train.latest_checkpoint("mytfemodel/")
if latest_cpkt:
    print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint.restore(latest_cpkt)


def loss_mask(real, pred):
    '''使用Softmax屏蔽计算损失'''
    mask = 1 - np.equal(real, 0)  # 批次中被补0的序列不参与计算loss
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


def all_loss(encoder, decoder, img_tensor, target):
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([word_index['<start>']] * target.shape[0], 1)
    feature = encoder(img_tensor)  # (batch_size,49,256)

    for i in range(1, target.shape[1]):
        predictions, hidden, _ = decoder(dec_input, feature, hidden)
        loss += loss_mask(target[:, i], predictions)

        dec_input = tf.expand_dims(target[:, i], 1)
    return loss


grad = tfe.implicit_gradients(all_loss)


# 实现单步训练过程
def train_one_epoch(encoder, decoder, optimizer, step_counter, dataset, epoch):
    total_loss = 0
    for (step, (img_tensor, target)) in enumerate(dataset):
        loss = 0

        optimizer.apply_gradients(grad(encoder, decoder, img_tensor, target), step_counter)
        loss = all_loss(encoder, decoder, img_tensor, target)

        total_loss += (loss / int(target.shape[1]))
        if step % 5 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         step,
                                                         loss.numpy() / int(target.shape[1])))
    print("step", step)
    return total_loss / (step + 1)


# 训练模型
loss_plot = []
EPOCHS = 50

for epoch in range(EPOCHS):
    start = time.time()
    total_loss = train_one_epoch(dataset=dataset, epoch=epoch, **model_objects)  # 训练一次

    loss_plot.append(total_loss)  # 保存loss

    print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss))
    checkpoint.save(checkpoint_prefix)
    print('Train time for epoch #%d (step %d): %f' %
          (checkpoint.save_counter.numpy(), checkpoint.step_counter.numpy(), time.time() - start))

#
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()


def evaluate(encoder, decoder, optimizer, step_counter, image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)
    size = [224, 224]

    def load_image(image_path):
        img = tf.read_file(PATH + image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, image_path

    from tensorflow.python.keras.applications.resnet import ResNet50

    image_model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
                           , include_top=False)  # 创建ResNet网络

    new_input = image_model.input
    hidden_layer = image_model.layers[-2].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    #    print(step_counter.numpy())
    dec_input = tf.expand_dims([word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        print(predictions.get_shape())

        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
        result.append(index_word[predicted_id])

        print(predicted_id)

        if index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(PATH + image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        # print(len(attention_plot[l]),attention_plot[l])
        temp_att = np.resize(attention_plot[l], (7, 7))
        ax = fig.add_subplot(len_result // 2, len_result // 2 + len_result % 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.4, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))

image = img_name_val[rid]
real_caption = ' '.join([index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image=image, **model_objects)

print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)
## opening the image
img = Image.open(PATH + img_name_val[rid])
plt.imshow(img)
plt.axis('off')
plt.show()

