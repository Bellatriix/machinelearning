# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'KNNalgorithm.py'
@Time : 2020/02/03
@Desc : 'K-NN algorithm'
'''

import numpy as np

class KNN:

    '''
    @Desc : 'KNN algorithm classify'
    @Parameters :
        inX - test set
        dataSet - train set
        labels - label of train set
        k - KNN algorithm's parameter, select the first k points
    @Returns :
        sotredClassCount[0][0] - result of classify
    @Time : 2020/1/31
    '''
    def classifyKNN(inX, dataSet, labels, k):
        #KNN本质为计算测试集与训练集各个点之间的欧氏距离，故用矩阵进行同时计算以减少计算量
        #获取训练集的行数。shape可以得到矩阵的行列数，不写[]返回行列数(rows,cols)，0返回行数
        dataSetSize = dataSet.shape[0]

        #将测试集复制，行数为训练集的行数，列数为1，复制后减去训练集
        #tile函数可以按行列(x,y)进行复制
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

        #将减去后的数据平方计算
        sqDiffMat = diffMat ** 2

        #平方后的数据按行求和
        sqDistances = sqDiffMat.sum(axis=1)

        #求和后的数据开方
        distances = sqDistances ** 0.5

        #返回从小到大排列的索引值
        sortedDistIndices = distances.argsort()

        #记录类别次数的字典
        classCount = {}
        for i in range(k):
            #获取元素类别
            votelabel = labels[sortedDistIndices[i]]
            #计算次数。dict.get(key,default)，返回指定键的值，若该值不存在则返回默认值
            classCount[votelabel] = classCount.get(votelabel, 0) + 1

        #获取字典中，值最大所对应的标签
        #classCount = sorted(classCount.items(),key=lambda kv:(kv[1],kv[0]))
        #print(max(classCount, key=classCount.get))

        return max(classCount, key=classCount.get)