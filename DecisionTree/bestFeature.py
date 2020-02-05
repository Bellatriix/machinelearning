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
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

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

"""
函数说明:获取决策树叶子结点的数目
 
Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def getNumLeafs(myTree):
    numLeafs = 0                                                #初始化叶子
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

"""
函数说明:获取决策树的层数
 
Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def getTreeDepth(myTree):
    maxDepth = 0                                                #初始化决策树深度
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth            #更新层数
    return maxDepth

"""
函数说明:绘制结点
 
Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)        #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

"""
函数说明:标注有向边属性值
 
Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                            #计算标注位置
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

"""
函数说明:绘制决策树
 
Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        #设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)                                                          #获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                                                            #获取决策树层数
    firstStr = next(iter(myTree))                                                            #下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    #标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

"""
函数说明:创建绘制面板
 
Parameters:
    inTree - 决策树(字典)
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                                                    #创建fig
    fig.clf()                                                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                                            #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                                            #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                #x偏移
    plotTree(inTree, (0.5,1.0), '')                                                            #绘制决策树
    plt.show()

'''
@Desc : '使用决策树分类'
@Parameter :
    'inputTree' - '决策树'
    'testVec' - '测试数据'
@:returns :
    'result' - '分类结果'
@Time : '2020/02/05'
'''
def classify(inputTree, testVec):
    treeDataFrame = pd.DataFrame(inputTree)

    result = treeDataFrame.iloc[testVec[0]].iloc[0]

    if isinstance(result, dict):
        del(testVec[0])
        result = classify(result, testVec)

    return result

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    #calcShannonEnt(dataSet)
    #bestFeature = chooseBestFeature(dataSet)
    #print('最优特征为%d' % bestFeature)
    #t = splitDataSet(dataSet, 0, 0)
    #print(t)
    featureLabels = []
    decisionTree = createTree(dataSet, labels, featureLabels)
    testVec = [0,1]
    result = classify(decisionTree, testVec)
    print(result)
    #print(decisionTree)
    #createPlot(decisionTree)