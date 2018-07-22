#coding=utf8
from __future__ import print_function
from __future__ import division
import util
from util import Data
import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    #output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    output = inputs * tf.reshape(alphas, [-1, sequence_length, 1])
    if not return_alphas:
        return output
    else:
        return output, alphas

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
        self.X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel],name='inputX')
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

    def buildCNN(self):
        '''
        增加了层数,增加了batch normalization,改用了奇数大小的cnn核.
        '''
        conv = tf.layers.conv2d(self.X,filters=32,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        conv = tf.layers.conv2d(activation,filters=32,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(2,2), padding='valid')
        #dropout = tf.nn.dropout(pool, self.keep_prob)

        conv = tf.layers.conv2d(pool,filters=64,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        conv = tf.layers.conv2d(activation,filters=64,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(2,2), padding='valid')
        #dropout = tf.nn.dropout(pool, self.keep_prob)

        conv = tf.layers.conv2d(pool,filters=128,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        conv = tf.layers.conv2d(activation,filters=128,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(2,2), padding='valid')

        flatten = tf.reshape(pool, [-1, pool.shape[1].value*pool.shape[2].value*pool.shape[3].value])
        dense = tf.layers.dense(flatten,units = 512)
        dense = tf.nn.relu(dense)
        dropout = tf.nn.dropout(dense, self.keep_prob)
        self.features=dense
        dense = tf.layers.dense(dropout, units=self.CLASSNUM)
        self.fc3 = dense
        self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))  

    def buildCNN3(self):
        '''
        增加了层数,增加了batch normalization,改用了奇数大小的cnn核.
        '''
        with tf.variable_scope('hidden1',regularizer=self.regularizer,initializer=self.init):
            conv = tf.layers.conv2d(self.X,filters=16,kernel_size=[3,3],strides=(1,1),padding='valid')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(2,2), padding='valid')
            #dropout = tf.nn.dropout(pool, self.keep_prob)#实测dropout加的过多会使得难以收敛或者不收敛,且bn一定程度上可以替代dropout.
            #hidden1 = dropout
            hidden1 = pool
        print(hidden1.shape)
        with tf.variable_scope('hidden2',regularizer=self.regularizer,initializer=self.init):
            conv = tf.layers.conv2d(hidden1,filters=32,kernel_size=[3,3],strides=(1,1),padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(2,2), padding='same')
            #dropout = tf.nn.dropout(pool, self.keep_prob)
            #hidden2 = dropout
            hidden2 = pool
        print(hidden2.shape)
        with tf.variable_scope('hidden3',regularizer=self.regularizer,initializer=self.init):
            conv = tf.layers.conv2d(hidden2,filters=128,kernel_size=[3,3],strides=(1,1),padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[3, 3], strides=(2,2), padding='valid')
            #dropout = tf.nn.dropout(pool, self.keep_prob)
            #hidden3 = dropout
            hidden3 = pool
        print(hidden3.shape)
        with tf.variable_scope('hidden3_1',regularizer=self.regularizer,initializer=self.init):
            conv = tf.layers.conv2d(hidden3,filters=256,kernel_size=[3,3],strides=(1,1),padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(1,1), padding='valid')
            #dropout = tf.nn.dropout(pool, self.keep_prob)
            #hidden3_1 = dropout
            hidden3_1 = pool
        print(hidden3_1.shape)
        flatten = tf.reshape(hidden3_1, [-1, hidden3_1.shape[1].value*hidden3_1.shape[2].value*hidden3_1.shape[3].value])

        print(flatten.shape)
        with tf.variable_scope('hidden4',regularizer=self.regularizer,initializer=self.init):
            dense = tf.layers.dense(flatten,units = 1024)
            dense = tf.layers.batch_normalization(dense,training = self.is_training)
            dense = tf.nn.relu(dense)
            dropout = tf.nn.dropout(dense, self.keep_prob)
            hidden4 = dropout
        with tf.variable_scope('hidden5',regularizer=self.regularizer,initializer=self.init):
            dense = tf.layers.dense(hidden4,units = 1024)
            dense = tf.layers.batch_normalization(dense,training = self.is_training)
            dense = tf.nn.relu(dense)
            dropout = tf.nn.dropout(dense, self.keep_prob)
            hidden5 = dropout
        self.features = hidden5
        with tf.variable_scope('output',initializer=self.init):
            dense = tf.layers.dense(hidden5, units=self.CLASSNUM)
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
            #if e != 0 and (e %(num_epochs-1) == 0 or e%20==0): #每迭代一定次数保存下模型
            if e in [45,46,47,48,49]:
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
    
