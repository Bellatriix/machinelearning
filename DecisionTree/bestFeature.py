# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'bestFeature.py'
@Time : '2020/02/04'
@Desc : 'Decision Tree algorithm'
'''

import pandas as pd
import numpy as np
from math import log

'''
@Desc : '创建数据集'
@Parameter : 'none'
@Returns : 
    'dataSet' - '数据集' 
    'labels' - '分类标签'
@Time : '2020/02/04'
'''
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']

    return dataSet, labels

'''
@Desc : '计算给定数据集的香农熵，计算公式中，只要最终结果的分类，即yes，no'
@Parameter : 
    'dataSet' - '数据集'
@Returns :
    'shannonEnt' - '香农熵（经验熵）'
@Time : '2020/02/04'
'''
def calcShannonEnt(dataSet):
    #将数据集转为dataframe
    dataFrameSet = pd.DataFrame(dataSet)

    #取出最后一列，即结果列
    labelCountsSeries = dataFrameSet.iloc[:, -1]
    #结果计数
    labelCounts = labelCountsSeries.value_counts()
    #统计总数
    numEntires = len(labelCountsSeries)

    shannonEnt = 0.0
    for i in labelCounts:
        prob = float(i/numEntires)
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

'''
@Desc : '划分数据集'
@Parameter : 
    'dataSet' - '原始数据集'
    'column' - '需要分组的列'
@Returns :
    'splitList' - '划分后的list，元素为每组数据的list'
'''
def splitDataSet(dataSet, column, value = None):
    if value is None:
        #将数据集转为dataframe
        dataFrameSet = pd.DataFrame(dataSet)
        #获取列的唯一值列表
        indexList = dataFrameSet[column].value_counts().index
        #返回的划分列表
        splitList = []

        for i in indexList:
            group = dataFrameSet[dataFrameSet[column].isin([i])]
            #dataframe变为list，需要先用numpy变为ndarray，再变为list
            groupList = np.array(group).tolist()

            splitList.append(groupList)

        return splitList
    else:
        retDataSet = []                                        #创建返回的数据集列表
        for featVec in dataSet:                             #遍历数据集
            if featVec[column] == value:
                reducedFeatVec = featVec[:column]                #去掉axis特征
                reducedFeatVec.extend(featVec[column+1:])     #将符合条件的添加到返回的数据集
                retDataSet.append(reducedFeatVec)
        return retDataSet                                      #返回划分后的数据集

'''
@Desc : '选择最优特征'
@Parameter :
    'dataSet' - '数据集'
@Returns : 
    'bestFeature' - '最优特征值'
@Time : ''2020/02/04
'''
def chooseBestFeature(dataSet):
    #计算香农熵
    shannonEnt = calcShannonEnt(dataSet)
    #特征数量
    numFeatures = len(dataSet[0]) - 1
    #统计总数
    numEntires = len(dataSet)
    #最优信息增益
    bestInfoGain = 0.0
    #最优特征
    bestFeature = -1

    #遍历划分数据集，并计算每列的信息增益
    for i in range(numFeatures):
        splitList = splitDataSet(dataSet, i)

        #条件熵
        newEntropy = 0.0
        #对分组后的每一类数据分别算条件熵
        for j in splitList:
            newEntropy += calcShannonEnt(j) * len(j)/numEntires

        infoGain = shannonEnt - newEntropy
        #print('第%d个元素的信息增益为%f' % (i, infoGain))

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

'''
@Desc : '创建决策树'
@Parameter :
    'dataSet' - '训练集'
    'labels' - '分类标签'
    'featureLabels' - '存储选择的最优特征标签'
@Returns :
    'decisionTree' - '决策树'
@Time : '2020/02/05' 
'''
def createTree(dataSet, labels, featureLabels):
    #递归创建决策树，有两种结束情况。1.分类标签完全相同；2.分类特征不够用
    dataFrameSet = pd.DataFrame(dataSet)
    lastLabelSeries = dataFrameSet.iloc[:, -1]
    #分类标签完全相同
    if len(lastLabelSeries.value_counts()) == 1:
        return np.array(lastLabelSeries).tolist()[0]
    #特征已用完只剩结果，则返回分类标签中出现最多的
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return lastLabelSeries.value_counts().idxmax()

    #选择最优特征
    bestFeat = chooseBestFeature(dataSet)
    #最优特征的标签
    bestFeatLabel = labels[bestFeat]
    featureLabels.append(bestFeatLabel)
    decisionTree = {bestFeatLabel:{}}
    #删除已经使用的最有特征
    del(labels[bestFeat])

    #取得特征的所有取值情况
    featureSeries = dataFrameSet.iloc[:, bestFeat]
    uniqueVals = list(featureSeries.value_counts().index)

    #遍历特征，创建决策树
    for value in uniqueVals:
        #list深拷贝
        subLabels = labels[:]

        decisionTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featureLabels)

    return decisionTree

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    #calcShannonEnt(dataSet)
    #bestFeature = chooseBestFeature(dataSet)
    #print('最优特征为%d' % bestFeature)
    #t = splitDataSet(dataSet, 0, 0)
    #print(t)
    featureLabels = []
    decisionTree = createTree(dataSet, labels, featureLabels)
    print(decisionTree)