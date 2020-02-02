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

    #数据切片
    labels = datingData[3]
    dataSet = datingData.drop(3, axis=1)

    #将dataframe变为numpy数组
    returnMat = dataSet.values
    classLabelVector = labels.values

    return returnMat,classLabelVector

if __name__ == '__main__':
    filename = './datingTestSet.txt'
    returnMat, classLabelVector = file2matrix(filename)
    print(type(returnMat),type(classLabelVector))
