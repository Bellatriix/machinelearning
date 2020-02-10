# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'horseColic.py'
@Time : '2020/02/10'
@Desc : 'Gradient rise algorithm'
'''

import numpy as np
import random
from sklearn.linear_model import LogisticRegression

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

'''
@Desc : '分类函数'
@Parameters : 
    'inX' - '特征向量'
    'weights' - '回归系数'
@Return : '分类结果'
@Time : '2020/02/10'
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

'''
@Desc : '使用逻辑分类器预测'
@Parameter : '无'
@Return : '无'
@Time : '2020/02/10'
'''
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))

    trainWeights = gradAscent(np.array(trainingSet), trainingLabels, 500)

    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1

    errorRate = (float(errorCount) / numTestVec) * 100

    print("错误率：%.2f%%" % errorRate)

'''
@Desc : '使用sklearn逻辑分类器预测'
@Parameter : '无'
@Return : '无'
@Time : '2020/02/10'
'''
def colicTest1():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))

    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))

    classifier = LogisticRegression(solver='liblinear', max_iter=20).fit(trainingSet, trainingLabels)
    testAccurcy = classifier.score(testSet, testLabels) * 100

    print("正确率：%.f%%" % testAccurcy)

if __name__ == '__main__':
    colicTest1()