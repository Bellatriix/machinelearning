# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'gradientRising.py'
@Time : '2020/02/10'
@Desc : 'Gradient rise algorithm'
'''

import matplotlib.pyplot as plt
import numpy as np
import random

'''
@Desc : '加载数据'
@Parameter : '无'
@Returns : 
    'dataMat' - '数据列表'
    'labelMat' - '标签列表'
@Time : '2020/02/10'
'''
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./testSet.txt')

    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    fr.close()

    return dataMat, labelMat

'''
@Desc : '绘制数据集'
@Parameter : 
    'weights' - '权重参数矩阵'
@Return : '无'
@Time : '2020/02/10'
'''
def plotDataSet(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]

    #正样本
    xcord1 = []
    ycord1 = []
    #负样本
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=0.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=0.5)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

'''
@Desc : 'sigmoid函数'
@Parameter :
    'inX' - '数据'
@Return :
    'sigmoid函数'
@Time : '2020/02/10'
'''
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

'''
@Desc : '梯度上升算法改进'
@Parameters 
    'dataMatix' - '数据集'
    'classLabels' - '分类标签'
    'numIter' - '迭代次数'
@Return :
    'weights' - '求得权重数据（最优参数）'
@Time : '2020/02/10'
'''
def gradAscent(dataMatix, classLabels, numIter=150):
    m, n = np.shape(dataMatix)
    weights = np.ones(n)

    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #降低alaph，每次减小1/(j+i)
            alpha = 4/(1.0 + j + i) + 0.01
            #随机选择样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            #选择随机选取的样本计算h
            h = sigmoid(sum(dataMatix[dataIndex[randIndex]] * weights))
            #计算误差
            error = classLabels[dataIndex[randIndex]] - h
            #更新回归系数
            weights = weights + alpha * error * dataMatix[dataIndex[randIndex]]
            #删除已使用的样本
            del(dataIndex[randIndex])

    #返回权重数组
    return weights

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(np.array(dataMat), labelMat)
    plotDataSet(weights)