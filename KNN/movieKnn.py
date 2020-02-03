# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'movieKnn.py'
@Time : 2020/1/31
@Desc : 'K-NN algorithm practice'
'''

import numpy as np
from KNN.KNNalgorithm import KNN

'''
@Desc : 'create data set'
@Parameter : no
@Returns : 
    group - data set
    labels - classified label
@Time : 2020/1/31
'''
def createDataSet():
    #创建数据集，此处为四行两列，即特征，x
    group = np.array([[1,101], [5,89], [108,5], [115,8]])

    #创建对应标签，即结果，y
    labels = ['爱情片', '爱情片', '动作片', '动作片']

    return group,labels

#如果本文件直接作为脚本，则会运行此内容，作为包被impor不会运行
if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()

    #测试集
    test = [101,20]

    #KNN分类
    result = KNN.classifyKNN(test, group, labels, 3)
    print(result)