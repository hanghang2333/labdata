#coding=utf8
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
np.random.seed(1)#固定下来随机化shuffle的序列
def get_generator(featurewise_center=False, featurewise_std=False,
                  rotation=5, width_shift=0.05, height_shift=0.05,
                  zoom=[0.9, 1.1], horizontal=False, vertical=False):
    '''
    图片数据随机化生成器定义，具体含义参看keras文档
    '''
    datagen = ImageDataGenerator(
        featurewise_center=featurewise_center,
        featurewise_std_normalization=featurewise_std,
        rotation_range=rotation,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        zoom_range=zoom,
        horizontal_flip=horizontal,
        vertical_flip=vertical)
    return datagen

def shuffledata(X,Y,all_path):
    '''
    将X和Y进行随机化排序
    '''
    indices = list(range(len(Y))) # indices = the number of images in the source data set
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    all_path = all_path[indices]
    return X,Y,all_path

class Data():
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.length = len(X)
        self.index = 0

    def get_next_batch(self, batch_size):
        '''
        以batch_size大小来取数据
        '''
        while self.index+batch_size < self.length:
            returnX = self.X[self.index:self.index+batch_size]
            returnY = self.Y[self.index:self.index+batch_size]
            self.index = self.index + batch_size
            yield returnX,returnY