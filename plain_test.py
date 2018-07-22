import cv2,codecs,os
os.environ["CUDA_VISIBLE_DEVICES"]='0'#args.gpu_core'
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import dataset,image_process
from plain_model import resnet
X_train, X_test, y_train, y_test,index_label = dataset.get_data('fmnist')
num_classes = len(index_label)
print('numclass:',num_classes)
image_height,image_width,image_channel = 28,28,1
# 识别模型
g1 = tf.Graph()
with g1.as_default():
    sess1 = tf.Session()#全局load模型
    model1 = resnet(image_height,image_width,image_channel,0.5,num_classes)
    saver1 = tf.train.Saver()
    # 读取训练好的模型参数
    saver1.restore(sess1, 'logs/0619/49model')#0.924

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
    tpres = res1[0]#toupiao(res1)
    if tpres == label:
        dictr[dictridx] = [img_a,res1,score,acc]
        dictridx+=1
    else:
        dictw[dictwidx] = [img_a,res1,score,acc]
        dictwidx+=1
    #break
print(len(imglist),dictr,dictw)

''' 

    strw = str(label)+' '
    res1 = [str(k) for k in res1]
    strw += ' '.join(list(res1))
    strw +='\t'
    strwlist = []
    for idx,i in enumerate(score):
        nowstrw = ''
        nowscore = list(i)
        nowacc = list(acc[idx])
        nowscore = [str(k) for k in nowscore]
        nowacc = [str(k) for k in nowacc]
        nowstrw +=' '.join(list(nowscore))
        nowstrw +=' '.join(list(nowacc))
        strwlist.append(nowstrw)
    strwlist = '\t'.join(strwlist)
    strw+=strwlist
    strw +='\n'
    if tpres==label:
        #right
        write_file_r.write(strw)
    else:
        #wrong
        write_file_w.write(strw)
        
'''

print('start dump')
import pickle
# pickle a variable to a file
file1 = open('dictr_only.pkl', 'wb')
pickle.dump(dictr, file1)
file1.close()

file1 = open('dictw_only.pkl', 'wb')
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
