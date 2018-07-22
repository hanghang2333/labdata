import cv2,codecs,os
os.environ["CUDA_VISIBLE_DEVICES"]='0'#args.gpu_core'
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import dataset,image_process
from plain_model import resnet
#X_train, X_test, y_train, y_test,index_label = dataset.get_data('fmnist')
num_classes = 10#len(index_label)
print('numclass:',num_classes)
image_height,image_width,image_channel = 28,28,1
# 识别模型
g1 = tf.Graph()

var_list_name = None
var_add = [[],[],[],[],[]]
with g6.as_default():
    index = 0
    print('load model...')
    sess6 = tf.Session()#全局load模型
    model6 = resnet(image_height,image_width,image_channel,0.5,num_classes)
    saver6 = tf.train.Saver()
    # 读取训练好的模型参数
    saver6.restore(sess1, 'logs/0720/45model')#0.924
    print('load done...')
    var_list = tf.trainable_variables()
    var_list_name = [i.name for i in var_list]
    for name in var_list_name:
        var_add[index].append(sess6.run(name))
    '''
    print(var_list[0])
    print(var_list[0].name)
    params1 = sess1.run('hidden1/conv2d/kernel:0')
    #print(params1)
    #print('sssssss')
    sess1.run(tf.assign(var_list[0],np.zeros((3,3,1,16))))
    params1 = sess1.run('hidden1/conv2d/kernel:0')
    #print(params1)
    '''
with g2.as_default():
    index = 1
    print('load model...')
    sess2 = tf.Session()#全局load模型
    model2 = resnet(image_height,image_width,image_channel,0.5,num_classes)
    saver2 = tf.train.Saver()
    # 读取训练好的模型参数
    saver2.restore(sess2, 'logs/0720/46model')#0.924
    print('load done...')
    var_list = tf.trainable_variables()
    var_list_name = [i.name for i in var_list]
    for name in var_list_name:
        var_add[index].append(sess2.run(name))

with g3.as_default():
    index = 2
    print('load model...')
    sess3 = tf.Session()#全局load模型
    model3 = resnet(image_height,image_width,image_channel,0.5,num_classes)
    saver3 = tf.train.Saver()
    # 读取训练好的模型参数
    saver3.restore(sess3, 'logs/0720/47model')#0.924
    print('load done...')
    var_list = tf.trainable_variables()
    var_list_name = [i.name for i in var_list]
    for name in var_list_name:
        var_add[index].append(sess3.run(name))

with g4.as_default():
    index = 3
    print('load model...')
    sess4 = tf.Session()#全局load模型
    model4 = resnet(image_height,image_width,image_channel,0.5,num_classes)
    saver4 = tf.train.Saver()
    # 读取训练好的模型参数
    saver4.restore(sess4, 'logs/0720/48model')#0.924
    print('load done...')
    var_list = tf.trainable_variables()
    var_list_name = [i.name for i in var_list]
    for name in var_list_name:
        var_add[index].append(sess4.run(name))

with g5.as_default():
    index = 4
    print('load model...')
    sess5 = tf.Session()#全局load模型
    model5 = resnet(image_height,image_width,image_channel,0.5,num_classes)
    saver5 = tf.train.Saver()
    # 读取训练好的模型参数
    saver5.restore(sess5, 'logs/0720/49model')#0.924
    print('load done...')
    var_list = tf.trainable_variables()
    var_list_name = [i.name for i in var_list]
    for name in var_list_name:
        var_add[index].append(sess5.run(name))

with g1.as_default():
    index = 5
    print('load model...')
    sess1 = tf.Session()#全局load模型
    model1 = resnet(image_height,image_width,image_channel,0.5,num_classes)
    saver1 = tf.train.Saver()
    # 读取训练好的模型参数
    saver1.restore(sess1, 'logs/0720/45model')#0.924
    print('load done...')
    var_list = tf.trainable_variables()
    #calc mean
    mean = []
    for idx,i in enumerate(range(len(var_list_name))):
        tmp = (var[0][idx]+var[1][idx]+var[2][idx]+var[3][idx]+var[4][idx])/5
        sess1.run(tf.assign(var_list[idx],tmp))
    print('mean done')

def predict_result_cate(X):
    '''
    X:四维矩阵[图片个数,图片高度,图片宽度,通道数],已经经过了归一化
    return:预测结果列表
    '''
    with g1.as_default():
        res = model1.predict(sess1, X)
    return res
    def predict(self,sess,X):
        scoreval = sess.run(self.score,feed_dict={self.X:X,self.keep_prob:1,self.is_training:False})
        score1_acc = tf.reshape(tf.nn.softmax(scoreval),[-1,])
        #print(score1_acc.shape)
        res = sess.run(tf.argmax(scoreval,axis=1))
        acc = sess.run(score1_acc)
        return res,scoreval,acc

def toupiao(lis):
    t = dict()
    for i in lis:
        if i in t:
            t[i]+=1
        else:
            t[i]=1
    t = sorted(t.items(), key=lambda item:item[1], reverse=True)
    return t[0][0]

dictr,dictridx = dict(),0
dictw,dictwidx = dict(),0
imglist = X_test
aug = image_process.image_aug()
for idx,img in tqdm(enumerate(imglist)):
    img = img*255
    label = y_test[idx]
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
    img_a = np.array([img,img1,img2,img3,img4,img5,img6,img7,img8,img9,img10])
    img_a = img_a/255
    
    res = predict_result_cate(img_a)
    #print(res)
    res1,score,acc = res
    acc = np.reshape(acc,(11,10))
    #print(res1.shape,score.shape,acc.shape)
    res1 = list(res1)
    tpres = toupiao(res1)#res1[0]#toupiao(res1)
    if tpres == label:
        dictr[dictridx] = [img_a,res1,score,acc]
        dictridx+=1
    else:
        dictw[dictwidx] = [img_a,res1,score,acc]
        dictwidx+=1
    #break
print(len(imglist),dictr,dictw)

print('start dump')
import pickle
# pickle a variable to a file
file1 = open('dictr_save.pkl', 'wb')
pickle.dump(dictr, file1)
file1.close()

file1 = open('dictw_save.pkl', 'wb')
pickle.dump(dictw, file1)
file1.close()

'''
# reload a file to a variable
with open('dictr.pkl', 'rb') as file1:
    dictr =pickle.load(file1)
    print('dictr:',len(dictr))
with open('dictw.pkl', 'rb') as file1:
    dictw =pickle.load(file1)
    print('dictw:',len(dictw)) 
'''
