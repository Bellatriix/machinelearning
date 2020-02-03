# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'dateKnn.py'
@Time : '2020/02/02'
@Desc : 'KNN algorithm for dating website matching'
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties

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

'''
@Desc : '打开并解析数据文件datingTestSet.txt，前三列分别代表飞行里程，娱乐耗时百分比，每周冰淇淋消耗公升数量'
        '第四列为labels。前三列写为特征矩阵，第四列为labels向量’
@Parameter : 
    'filename' - '文件名'
@Returns : 
    'returnMat' - '特征矩阵'
    'classLabelVector' - '分类向量'
@Time : '2020/02/02'
'''
def file2matrix(filename):
    #用pandas读取文件数据
    datingData = pd.read_table(filename, header=None, sep='	')

    #获取最后一列的列名
    lastColumn = datingData.columns[len(datingData.columns)-1]

    #数据切片
    labels = datingData[lastColumn]
    dataSet = datingData.drop(lastColumn, axis=1)

    #将labels中的字符串变为对应数字，1为不喜欢，2为魅力一般，3为极具魅力
    labelsMapping = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    labels = labels.map(labelsMapping)

    #将dataframe变为numpy数组
    returnMat = dataSet.values
    classLabelVector = labels.values

    return returnMat,classLabelVector

'''
@Desc : '数据可视化'
@Parameter : 
    'datingDataMat' - '特征矩阵'
    'datingLabels' - '分类label'
@Returns : '无'
@Time : '2020/02/03'
'''
def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    #将画布分为两行两列
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)

    labelsColors = []
    for i in datingLabels:
        if i == 1:
            labelsColors.append('red')
        if i == 2:
            labelsColors.append('yellow')
        if i == 3:
            labelsColors.append('blue')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客里程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=labelsColors, s=15, alpha=0.5)
    #设置坐标轴和图标题，及字体大小和颜色
    axs0_title_text = axs[0][0].set_title(u'飞行常客里程与玩游戏所消耗的时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'飞行常客里程', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩游戏消耗时间', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客里程)、第三列(冰淇淋)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=labelsColors, s=15, alpha=0.5)
    #设置坐标轴和图标题，及字体大小和颜色
    axs0_title_text = axs[0][1].set_title(u'飞行常客里程与冰淇凌消耗占比', FontProperties=font)
    axs0_xlabel_text = axs[0][1].set_xlabel(u'飞行常客里程', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'冰淇淋消耗', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰淇凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=labelsColors, s=15, alpha=0.5)
    #设置坐标轴和图标题，及字体大小和颜色
    axs0_title_text = axs[1][0].set_title(u'玩游戏消耗时间与冰淇淋消耗占比', FontProperties=font)
    axs0_xlabel_text = axs[1][0].set_xlabel(u'玩游戏消耗时间', FontProperties=font)
    axs0_ylabel_text = axs[1][0].set_ylabel(u'冰淇淋消耗', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',  markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')

    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

'''
@Desc : '样本数据进行归一化'
@Parameter : 
    'dataSet' - '特征矩阵'
@Returns :
    'normalDataSet' - '归一化后的特征矩阵'
    'ranges' - '数据范围'
    'minVals' - '最小值'
@Time : '2020/02/02'
'''
def autoNorm(dataSet):
    #归一值 = (x-min) / (max - min)
    #min()返回所有元素最小值，min(0)返回每一列最小值，min(1)返回每行最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    #最大值和最小值的范围
    ranges = maxVals - minVals

    #获取dataset行数
    m = dataSet.shape[0]

    #原始值减去最小值
    normalDataSet = dataSet - np.tile(minVals, (m,1))
    #除 max-min 得到归一值
    normalDataSet = normalDataSet / np.tile(ranges, (m,1))

    return normalDataSet,ranges,minVals

'''
@Desc : '分类器测试函数'
@Parameter : 
    'filename' - '文件路径'
@Returns :
    'errorRate' - '错误率'
@Time : '2020/02/03'
'''
def datingClassTest(filename):
    returnMat, classLabelVector = file2matrix(filename)

    normalDataSet, ranges, minVals = autoNorm(returnMat)

    #数据的百分之十，个数
    testRatio = 0.10
    m = normalDataSet.shape[0]
    numTestVecs = int(m * testRatio)
    #错误个数
    errorCount = 0

    for i in range(numTestVecs):
        #前numTestVecs作为测试集，后 m-numTestVesc 作为训练集
        classifierResult = classifyKNN(normalDataSet[i, :], normalDataSet[numTestVecs:m, :], classLabelVector[numTestVecs:m], 10)

        #print("分类结果:%d\t真实类别:%d" % (classifierResult, classLabelVector[i]))
        if classifierResult != classLabelVector[i]:
            errorCount += 1

    #错误率
    errorRate = float(errorCount/numTestVecs) * 100
    #print("错误率:%f%%" %(errorRate))
    return errorRate

'''
@Desc : '测试例子'
@Parameter : '无'
@Returns : '测试结果'
@Time : '2020/02/03'
'''
def classifyPerson():
    #三项特征输入
    ffMiles = float(input("每年飞行里程数："))
    precentTats = float(input("玩游戏消耗时间占比："))
    iceCream = float(input("每周消耗冰淇淋公升数："))

    filename = './datingTestSet.txt'

    #处理训练数据
    datingDataMat, datingLabels = file2matrix(filename)
    #训练数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)

    #生成测试集
    inX = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    normInX = (inX - minVals) / ranges

    #返回分类结果
    classifyResult = classifyKNN(normInX, normMat, datingLabels, 3)

    #结果映射
    resultList = ['讨厌', '有些喜欢', '非常喜欢']

    print("你可能%s这个人" % (resultList[classifyResult - 1]))

if __name__ == '__main__':
    filename = './datingTestSet.txt'
    #returnMat, classLabelVector = file2matrix(filename)
    #showdatas(returnMat, classLabelVector)
    #normalDataSet, ranges, minVals = autoNorm(returnMat)
    #datingClassTest(filename)
    classifyPerson()