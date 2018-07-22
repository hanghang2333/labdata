import pickle
import numpy as np
from sklearn.model_selection import train_test_split
# reload a file to a variable
with open('dictr1.pkl', 'rb') as file1:
    dictr =pickle.load(file1)
    print('dictr:',len(dictr))
with open('dictw1.pkl', 'rb') as file1:
    dictw =pickle.load(file1)
    print('dictw:',len(dictw)) 
print('load data done')
X = []
label = []
for i in range(len(dictw)):
    X.append(dictw[i][3])
    label.append(0)
for i in range(len(dictw)):
    X.append(dictr[i][3])
    label.append(1)
X = np.array(X)
X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
label = np.array(label)
print(X.shape,label.shape)

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.1, random_state=42)


import codecs
import argparse
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split,KFold
from sklearn.utils import shuffle
import os,sys
from model_uncer_mod import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_core',type=str,default='0',help='use which gpu core')
parser.add_argument('--logs',type=str,default='logs/0625',help='logs path')
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
# 超参数
num_classes = 2
image_height,image_width,image_channel = 11,10,1

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
os.environ["CUDA_VISIBLE_DEVICES"]='0'#args.gpu_core'
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    print('Train: ',len(X_train))
    print('Test: ',len(X_test))
    model = resnet(image_height, image_width, image_channel, 0.5, num_classes)
    res = model.train(sess, X_train, y_train, X_test, y_test,num_epochs=1000,
    num_count=5,logs=args.logs)
    print(res)
