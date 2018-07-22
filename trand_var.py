import numpy as np
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats
# reload a file to a variable
with open('dictr_only.pkl', 'rb') as file1:
    dictr =pickle.load(file1)
    print('dictr:',len(dictr))
with open('dictw_only.pkl', 'rb') as file1:
    dictw =pickle.load(file1)
    print('dictw:',len(dictw)) 
print('load data done')
rightX = []
wrongX = []
right_label = []
wrong_label = []

for i in range(len(dictw)):
    wrongX.append(dictw[i][3])
    wrong_label.append(0)
for i in range(len(dictw)):
    rightX.append(dictr[i][3])
    right_label.append(1)
rightX = np.array(rightX)
wrongX = np.array(wrongX)
right_label = np.array(right_label)
wrong_label = np.array(wrong_label)
print(rightX.shape,right_label.shape)

#5.分析输出概率分布wassian散度
sample_wrongX = wrongX[0:len(wrongX)]
sample_rightX = rightX[0:len(wrongX)]
sample_rightX_wrongX = [sample_rightX,sample_wrongX]
import scipy.linalg
def wasserstein1(mu_x, cov_x, mu_y, cov_y):
    mu_diff = np.linalg.norm(mu_x - mu_y)
    trace_x = np.trace(cov_x)
    trace_y = np.trace(cov_y)
    sqrt_x = scipy.linalg.sqrtm(cov_x)
    sqrt_y = scipy.linalg.sqrtm(cov_y)
    mat = np.dot(np.dot(sqrt_x, sqrt_y), sqrt_x)
    return np.sqrt(mu_diff + trace_x + trace_y - 2*np.trace(scipy.linalg.sqrtm(mat)))

for ii in range(len(sample_rightX_wrongX)):
    plt.figure(ii)
    point = []
    for i in range(len(sample_rightX_wrongX[ii])):
        tmplabel = sample_rightX_wrongX[ii][i]
        tmp_kl = []
        for k in range(tmplabel.shape[0]):
            for k_2 in range(k+1,tmplabel.shape[1]):
                cov1 = 2*np.eye(10)
                cov2 = 4*np.eye(10)
                dr = wasserstein1(tmplabel[k],cov1,tmplabel[k_2],cov2)
                tmp_kl.append(dr)
        tmp_kl_mean = sum(tmp_kl)*1.0/len(tmp_kl)
        point.append(tmp_kl_mean)
    print(len(point),point[0:10])
    import seaborn as sns
    sns.set(color_codes=True)
    sns.distplot(np.array(point),bins=180)
    plt.savefig('gen_image/'+str(ii+12)+'.jpg')

#5.分析输出概率分布JS散度
#sample_wrongX = wrongX[0:len(wrongX)]
#sample_rightX = rightX[0:len(wrongX)]
#sample_rightX_wrongX = [sample_rightX,sample_wrongX]
#for ii in range(len(sample_rightX_wrongX)):
#    plt.figure(ii)
#    point = []
#    for i in range(len(sample_rightX_wrongX[ii])):
#        tmplabel = sample_rightX_wrongX[ii][i]
#        tmp_kl = []
#        for k in range(tmplabel.shape[0]):
#            for k_2 in range(k+1,tmplabel.shape[1]):
#                three = (tmplabel[k]+tmplabel[k_2])/2
#                dr = scipy.stats.entropy(tmplabel[k],three)+scipy.stats.entropy(tmplabel[k_2],three)
#                tmp_kl.append(dr)
#        tmp_kl_mean = sum(tmp_kl)*1.0/len(tmp_kl)
#        point.append(tmp_kl_mean)
#    print(len(point),point[0:10])
#    import seaborn as sns
#    sns.set(color_codes=True)
#    sns.distplot(np.array(point),bins=180)
#    plt.savefig('gen_image/'+str(ii+10)+'.jpg')
#4.分析输出概率分布KL散度
#sample_wrongX = wrongX[0:len(wrongX)]
#sample_rightX = rightX[0:len(wrongX)]
#sample_rightX_wrongX = [sample_rightX,sample_wrongX]
#for ii in range(len(sample_rightX_wrongX)):
#    plt.figure(ii)
#    point = []
#    for i in range(len(sample_rightX_wrongX[ii])):
#        tmplabel = sample_rightX_wrongX[ii][i]
#        tmp_kl = []
#        for k in range(tmplabel.shape[0]):
#            for k_2 in range(k+1,tmplabel.shape[1]):
#                tmp_kl.append(scipy.stats.entropy(tmplabel[k],tmplabel[k_2]))
#        tmp_kl_mean = sum(tmp_kl)*1.0/len(tmp_kl)
#        point.append(tmp_kl_mean)
#    print(len(point),point[0:10])
#    import seaborn as sns
#    sns.set(color_codes=True)
#    sns.distplot(np.array(point),bins=180)
#    plt.savefig('gen_image/'+str(ii+8)+'.jpg')

#3.分析输出概率分布方差
#思路：对每一个case，计算出其N张图片的概率分布和，而后计算出方差，为一个数值，而后将所有的画在图上。
#sample_wrongX = wrongX[0:len(wrongX)]
#sample_rightX = rightX[0:len(wrongX)]
#sample_rightX_wrongX = [sample_rightX,sample_wrongX]
#for ii in range(len(sample_rightX_wrongX)):
#    plt.figure(ii)
#    point = []
#    for i in range(len(sample_rightX_wrongX[ii])):
#        tmplabel = sample_rightX_wrongX[ii][i][0:1]
#        tmplabel_count = np.sum(tmplabel,axis=0)
#        tmplabel_count = tmplabel_count/len(tmplabel)
#        tmplabel_out = np.var(tmplabel_count)
#        point.append(tmplabel_out)
#    print(len(point),point[0:10])
#    import seaborn as sns
#    sns.set(color_codes=True)
#    sns.distplot(np.array(point),bins=80)
#    plt.savefig('gen_image/'+str(ii+6)+'.jpg')
#2.分析输出概率分布
#思路：正负样例各抽10个，因为分析每个样本的输出概率分布实在困难，所以这里求11个图片的加和。
#那么就和上面一样啦哈
# sample_rightX = rightX[0:36]
# sample_wrongX = wrongX[0:36]
# sample_rightX_wrongX = [sample_rightX,sample_wrongX]
# for ii in range(len(sample_rightX_wrongX)):
#     plt.figure(ii)
#     for i in range(36):
#         plt.subplot(6,6,i+1)
#         tmplabel = sample_rightX_wrongX[ii][i]

#         tmplabel_count = np.sum(tmplabel,axis=0)
#         tmplabel_out = list(tmplabel_count)
#         plt.pie(tmplabel_out)
#     plt.savefig('gen_image/'+str(ii+2)+'.jpg')


#1.分析标签分布
#思路：正负样例各抽10个，将他们的投票的标签分别画在一张图上。
#预期的结果是正例的标签分布要更集中一些，而分错的那些往往11张图片自己都统一不了意见这样。
# sample_rightX = rightX[0:36]
# sample_wrongX = wrongX[0:36]
# sample_rightX_wrongX = [sample_rightX,sample_wrongX]

# for ii in range(len(sample_rightX_wrongX)):
#     plt.figure(ii)
#     for i in range(36):
#         plt.subplot(6,6,i+1)
#         tmplabel = sample_rightX_wrongX[ii][i]
#         tmplabel_count = dict()
#         for k in tmplabel:
#             if k in tmplabel_count:
#                 tmplabel_count[k] +=1
#             else:
#                 tmplabel_count[k]=1
#         tmplabel_out = []
#         for k in tmplabel_count:
#             tmplabel_out.append(tmplabel_count[k])
#         plt.pie(tmplabel_out)
#     plt.savefig('gen_image/'+str(ii)+'.jpg')





