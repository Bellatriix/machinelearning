# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'sklearnDigitRecognition.py'
@Time : '2020/02/04'
@Desc : 'sklearn handwritten digit recognition'
'''

import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

'''
@Desc : '将32*32的二进制图像，转化为1*1024向量'
@Parameter : 
    'filename' - '文件路径'
@Returns :
    'returnVector' - '1*1024向量'
@Time : '2020/02/04'
'''


def img2vector(filename):
    returnVector = np.zeros((1, 1024))

    fr = open(filename)

    for i in range(32):
        # 读取一行数据
        lineStr = fr.readline()
        # 每一行的前32个数据放入向量中
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])

    return returnVector


'''
@Desc : '手写数字分类测试'
@Parameter : 'none'
@Returns : 'none'
@Time : '2020/02/04'
'''


def handwritingClassTest():
    # 测试集Labels
    hwLabels = []
    # 读取训练集下的文件名
    trainingFileList = listdir('./trainingDigits')
    # 文件个数
    m = len(trainingFileList)
    # 初始化训练矩阵
    trainingMat = np.zeros((m, 1024))
    # 从文件名解析除训练集类别
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将得到的结果加入labels
        hwLabels.append(classNumber)
        # 将每个文件生成的矩阵保存到Mat中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))

    # 构建KNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型，trainingMat为训练矩阵，hwlabels为标签
    neigh.fit(trainingMat, hwLabels)

    # 获取测试集
    testFileList = listdir('./testDigits')
    # 错误个数
    errorCount = 0.0
    # 测试集数量
    mTest = len(testFileList)
    # 对测试集进行分类测试
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        # 获取测试集的向量
        vectorTest = img2vector('testDigits/%s' % (fileNameStr))

        # 获得预测结果
        classifierResult = neigh.predict(vectorTest)
        print('分类结果：%d，真实结果：%d' % (classifierResult, classNumber))
        if classifierResult != classNumber:
            errorCount += 1.0

    print("一共错了%d个，错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()
