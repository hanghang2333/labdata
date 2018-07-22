#coding=utf8
from __future__ import print_function
from __future__ import division
import util
from util import Data
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    centers_weight = tf.get_variable('centers_weight', [1,len_features], dtype=tf.float32,
        initializer=tf.ones_initializer(), trainable=True)
    centers_weight = tf.nn.softmax(centers_weight)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])
    
    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.div(tf.nn.l2_loss(tf.multiply(centers_weight,features - centers_batch)),int(len_features))
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = tf.multiply(centers_weight,tf.multiply(centers_weight,features - centers_batch))#centers_batch - features
    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    return loss,centers_update_op
class resnet(object):
    '''
    模型初始化定义
    '''
    def __init__(self, image_height, image_width, image_channel, keep_prob, classNum):
        self.X = tf.placeholder(tf.float32, [None,image_height],name='inputX')
        self.y = tf.placeholder(tf.int64, [None],name='inputY')
        #self.bn = tf.placeholder(tf.bool,name='batchnorm')
        self.keep_prob_train = keep_prob
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.CLASSNUM = classNum
        #建立CNN
        self.init = tf.truncated_normal_initializer(0.0,0.01)#参数初始化方式
        self.regularizer = tf.contrib.layers.l2_regularizer(0.0)#L2正则,暂时保留
        #self.deepnn()
        self.buildCNN3()
        self.score = self.fc3
        self.probabi = self.probability() #模型输出结果的准确度.没有传参,但依赖于self.score值.
        # 损失函数定义
        with tf.variable_scope('loss_scope'):
            self.centerloss,self.centers_update_op = get_center_loss(self.features,self.y,0.5,self.CLASSNUM)
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.score)#+0.0*self.centerloss+self.regularization_loss
        # tf.summary.scalar('loss',self.loss)
        # 优化器
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,self.centers_update_op)
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)        
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(self.loss)
        # 准确度定义
        with tf.variable_scope('accuracy_scope'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(self.y,[1,-1]),
                          tf.reshape(tf.argmax(self.score,axis=1),[1,-1])), tf.float32))
            self.accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(self.y,[1,-1]),
                          tf.reshape(tf.argmax(self.score,axis=1),[1,-1])), tf.float32))
        tf.summary.scalar('train_accuracy',self.accuracy)
        tf.summary.scalar('test_accuracy',self.accuracy_test)
        self.merged = tf.summary.merge_all()
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    def probability(self):
        #确性度定义(即根据网络输出给出网络自身认为本次输出正确的概率值是多少)
        with tf.variable_scope('probabi_scope'):
            acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score),axis=1),[-1,1])
            probabi = acc*100
            probabi = tf.cast(probabi,tf.int64)
        return probabi

    def buildCNN3(self):
        flatten = self.X
        print('flatten:',flatten.shape)
        with tf.variable_scope('hidden4',regularizer=self.regularizer,initializer=self.init):
            dense = tf.layers.dense(flatten,units = 16)
            dense = tf.layers.batch_normalization(dense,training = self.is_training)
            dense = tf.nn.relu(dense)
            dropout = tf.nn.dropout(dense, self.keep_prob)
            hidden4 = dropout
        self.features = hidden4
        with tf.variable_scope('output',initializer=self.init):
            dense = tf.layers.dense(hidden4, units=self.CLASSNUM)
            dense = tf.layers.batch_normalization(dense,training = self.is_training)
            self.fc3 = dense
            self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))   

    
    def train(self, sess, X_train, Y_train, X_test, Y_test,num_epochs, num_count,logs):
        res = 0.0
        summary_writer = tf.summary.FileWriter(logs+'/train',sess.graph)
        summary_writer1 = tf.summary.FileWriter(logs+'/test',sess.graph)
        sess.run(self.init_op)
        datagen = util.get_generator()
        # 模型持久化器
        saver = tf.train.Saver()
        batch_size = 128
        print('batch_size:', batch_size)
        # 开始迭代
        for e in range(num_epochs):
            yieldData = Data(X_train,Y_train)
            print('Epoch', e)
            batches = 0
            if e != 0 and (e %(num_epochs-1) == 0 or e%20==0): #每迭代一定次数保存下模型
                saver.save(sess, logs+'/'+str(e)+'model')
            #使用原始数据进行迭代
            for i in range(1):
                for batch_X,batch_Y in yieldData.get_next_batch(batch_size):
                    _, lossval, scoreval = sess.run([self.train_op,self.loss, self.score],
                                                    feed_dict={self.X: batch_X, self.keep_prob:self.keep_prob_train, self.y:batch_Y,self.is_training:True})
            summary2, accuval, scoreval, lossval = sess.run([self.merged, self.accuracy_test, self.score,self.loss],
                                                 feed_dict={self.X: X_test, self.keep_prob:1, self.y:Y_test,self.is_training:False})
            print("Test accuracy:", accuval)
            print("Test loss:",lossval)
        return 0

    def predict(self,sess,X):
        scoreval = sess.run(self.score,feed_dict={self.X:X,self.keep_prob:1,self.is_training:False})
        score1_acc = tf.reshape(tf.nn.softmax(scoreval),[-1,])
        #print(score1_acc.shape)
        res = sess.run(tf.argmax(scoreval,axis=1))
        acc = sess.run(score1_acc)
        return res,scoreval,acc


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
    X.append(dictw[i][3][0])
    label.append(0)
for i in range(len(dictw)):
    X.append(dictr[i][3][0])
    label.append(1)
X = np.array(X)
X = np.reshape(X,(X.shape[0],X.shape[1]))
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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_core',type=str,default='0',help='use which gpu core')
parser.add_argument('--logs',type=str,default='logs/0625',help='logs path')
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
# 超参数
num_classes = 2
image_height,image_width,image_channel = 10,10,1

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
os.environ["CUDA_VISIBLE_DEVICES"]='0'#args.gpu_core'
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]))
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]))
    print('Train: ',X_train.shape)
    print('Test: ',len(X_test))
    model = resnet(image_height, image_width, image_channel, 0.5, num_classes)
    res = model.train(sess, X_train, y_train, X_test, y_test,num_epochs=300,
    num_count=5,logs=args.logs)
    print(res)
