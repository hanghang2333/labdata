import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def get_data(name='mnist'):
    if name=='mnist':
        return mnist()
    elif name=='notmnist':
        return notmnist()
    elif name=='cifar10':
        return cifar10()
    elif name=='cifar100':
        return cifar100()
    elif name=='fmnist':
        return fmnist()
    else:
        print('The name you provide is not exists!')

def mnist():
    import keras
    from keras.datasets import mnist as mnist1
    index_label={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
    (X_train, y_train), (X_test, y_test) = mnist1.load_data()
    X_train = np.reshape(X_train,(X_train.shape[0],28,28,1))
    X_test = np.reshape(X_test,(X_test.shape[0],28,28,1))
    print('info of data:',X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train/255, X_test/255, y_train, y_test,index_label

def fmnist():
    import keras
    from keras.datasets import fashion_mnist as mnist1
    index_label={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
    (X_train, y_train), (X_test, y_test) = mnist1.load_data()
    X_train = np.reshape(X_train,(X_train.shape[0],28,28,1))
    X_test = np.reshape(X_test,(X_test.shape[0],28,28,1))
    print('info of data:',X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train/255, X_test/255, y_train, y_test,index_label

def notmnist(path='notMNIST_small'):
    index_label={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}
    X = []
    y = []
    for index in index_label:
        imglist = os.listdir(os.path.join(path,index_label[index]))
        for imgf in imglist:
            imgname = os.path.join(path,index_label[index],imgf)
            im = Image.open(imgname)
            imarray = np.array(im)
            imarray = np.reshape(imarray,(imarray.shape[0],imarray.shape[1],1))
            X.append(imarray)
            y.append(index)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print('info of data:',X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train/255, X_test/255, y_train, y_test,index_label

def cifar10(path='cifar-10-batches-py'):
    import pickle
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    index_labelf = unpickle(os.path.join(path,'batches.meta'))[b'label_names']
    index_label = dict()
    for idx,i in enumerate(index_labelf):
        index_label[idx]=i
    databatch = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    X = []
    y = []
    for batch in databatch:
        dataf = unpickle(os.path.join(path,batch))
        tmpdata = dataf[b'data']
        tmpdata = np.reshape(tmpdata,(tmpdata.shape[0],32,32,3))
        X.extend(list(tmpdata))
        y.extend(dataf[b'labels'])
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print('info of data:',X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    return X_train/255, X_test/255, y_train, y_test,index_label

def cifar100(path='cifar-100-python'):
    import pickle
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    index_labelf = unpickle(os.path.join(path,'meta'))[b'fine_label_names']
    index_label = dict()
    for idx,i in enumerate(index_labelf):
        index_label[idx]=i
    databatch = ['train']
    X = []
    y = []
    for batch in databatch:
        dataf = unpickle(os.path.join(path,batch))
        tmpdata = dataf[b'data']
        tmpdata = np.reshape(tmpdata,(tmpdata.shape[0],32,32,3))
        X.extend(list(tmpdata))
        y.extend(dataf[b'fine_labels'])
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print('info of data:',X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train/255, X_test/255, y_train, y_test,index_label
