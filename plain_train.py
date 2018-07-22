import codecs
import argparse
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split,KFold
from sklearn.utils import shuffle
import os,sys
from plain_model import resnet
import dataset,util
import image_process

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_core',type=str,default='0',help='use which gpu core')
parser.add_argument('--logs',type=str,default='logs/0720',help='logs path')
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
# 超参数
num_classes = 10
image_height,image_width,image_channel = 28,28,1

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
os.environ["CUDA_VISIBLE_DEVICES"]='0'#args.gpu_core'
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    X_train, X_test, y_train, y_test,index_label=dataset.get_data('fmnist')
    #训练数据增强
    aug = image_process.image_aug()
    X_tra = []
    y_tra = []
    for idx,img in enumerate(X_train):
        img = img*255
        label = y_train[idx]
        img1 = aug.a_wflip(img)
        img2 = aug.a_rotate(img,10)
        img3 = aug.a_rotate(img,-10)
        img4 = aug.a_rotate(img,20)
        img5 = aug.a_rotate(img,-20)
        img6 = aug.a_shift(img,0.1,0.1)
        img7 = aug.a_shift(img,-0.1,-0.1)
        img8 = aug.a_zoom(img,0.9,0.9)
        img9 = aug.a_zoom(img,1.1,1.1)
        img10 = aug.a_crop(img,0.0,1.0,0.1,0.9)
        X_tra.extend([img,img1,img2,img3,img4,img5,img6,img7,img8,img9,img10])
        y_tra.extend([label for k in range(11)])
    X_train = np.array(X_tra)
    X_train = X_train/255
    y_train = np.array(y_tra)
    print('Train: ',len(X_train))
    print('Test: ',len(X_test))
    model = resnet(image_height, image_width, image_channel, 0.5, num_classes)
    res = model.train(sess, X_train, y_train, X_test, y_test,num_epochs=50,
    num_count=5,logs=args.logs)
    print(res)
