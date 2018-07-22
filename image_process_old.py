import cv2
import numpy as np
from keras.preprocessing import image as kimage
img = cv2.imread('test1.png',0)
print(img.shape)
img = np.reshape(img,(28,28,1))
def a_resize(img,height,width):
    img = img.copy()
    from skimage.transform import resize
    resized_image = resize(img,(height,width))
    resized_image = resized_image*255
    print(resized_image.shape)
    return resized_image
imgr = a_resize(img,50,50)
cv2.imwrite('testr.jpg',imgr)

def a_noise(img,point=255,num=100):
    img = img.copy()
    height,weight,channel = img.shape
    for i in range(num):
        x = np.random.randint(0,height)
        y = np.random.randint(0,weight)
        img[x ,y ,:] = point
    return img
img_n = a_noise(img)
print(img_n.shape)
cv2.imwrite('testn.jpg',img_n)

def a_hflip(img):
    img = img.copy()
    hflip = img[::-1,:,:]
    return hflip
img_n = a_hflip(img)
print(img_n.shape)
cv2.imwrite('testh.jpg',img_n)

def a_wflip(img):
    img = img.copy()
    hflip = img[:,::-1,:]
    return hflip
img_n = a_wflip(img)
print(img_n.shape)
cv2.imwrite('testw.jpg',img_n)

def a_rotate(img,theta,fill_mode='nearest',cval=0.):
    #theta为度数，正数是顺时针旋转
    x = img.copy()
    theta = np.pi / 180 * theta #逆时针旋转角度
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],\
          [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
    h, w = x.shape[0], x.shape[1]
    transform_matrix = kimage.transform_matrix_offset_center(rotation_matrix, h, w)
    x = kimage.apply_transform(x, transform_matrix, 2, fill_mode, cval)
    return x

img_n = a_rotate(img,180)
print(img_n.shape)
cv2.imwrite('testra.jpg',img_n)

def a_shift(img,hshift,wshift,fill_mode='nearest',cval=0.):
    #偏移正数是向左上方偏移
    x = img.copy()
    h, w = x.shape[0], x.shape[1] #读取图片的高和宽
    tx = hshift * h #高偏移大小，若不偏移可设为0，若向上偏移设为正数
    ty = wshift * w #宽偏移大小，若不偏移可设为0，若向左偏移设为正数
    translation_matrix = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])
    transform_matrix = translation_matrix  
    x = kimage.apply_transform(x, transform_matrix, 2, fill_mode, cval)
    return x

img_n = a_shift(img,-0.5,-0.5)
print(img_n.shape)
cv2.imwrite('tests.jpg',img_n)

def a_zoom(img,zx,zy,fill_mode='nearest', cval=0.):
    x = img.copy()
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[0], x.shape[1]
    transform_matrix = kimage.transform_matrix_offset_center(zoom_matrix, h, w) #保持中心坐标不改变
    x = kimage.apply_transform(x, transform_matrix, 2, fill_mode, cval)
    return x

img_n = a_zoom(img,0.5,0.5)
print(img_n.shape)
cv2.imwrite('testz.jpg',img_n)

def a_crop(img,hs,he,ws,we):
    x = img.copy()
    h,w = x.shape[0],x.shape[1]
    hs,he,ws,we = int(h*hs),int(h*he),int(ws*w),int(we*w)
    x = x[hs:he,ws:we,:]
    x = a_resize(x,h,w)
    return x

img_n = a_crop(img,0.2,0.8,0.2,0.8)
print(img_n.shape)
cv2.imwrite('testc.jpg',img_n)