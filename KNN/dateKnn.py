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
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties

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

if __name__ == '__main__':
    filename = './datingTestSet.txt'
    returnMat, classLabelVector = file2matrix(filename)
    showdatas(returnMat, classLabelVector)
