# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'speechFiltration.py'
@Time: '2020/02/06'
@Desc : 'Naive Bayes for speech filtering'
'''

import numpy as np

'''
@Desc : '创建实验样本'
@Parameter : 'none'
@Returns :
    'postingList' - '实验样本切分的词条'
    'classVec' - '分类标签'
@Time : '2020/02/06'
'''
def loadDataSet():
    #评论切分后的list，一行表示一句评论
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

'''
@Desc : '将数据列表整理成不重复的词汇表'
@Parameter :
    'dataSet' - '原始数据集'
@Return :
    'vocabSet' - '词汇表'
@Time : '2020/02/06'
'''
def createVocaList(dataSet):
    vocabSet = set([])
    for data in dataSet:
        #按位并运算
        vocabSet = vocabSet | set(data)

    return list(vocabSet)

'''
@Desc : '将数据根据词汇表变为向量'
@Parameters :
    'inputSet' - '切分后的词条'
    'vocabList' - '词汇表'
@Return :
    'returnVec' - '文档向量'
@Time : '2020/02/06'
'''
def word2Vec(inputSet, vocabList):
    returnVec = [0] * len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1

    return returnVec

'''
@Desc : '朴素贝叶斯训练函数'
@Parameters :
    'trainMatrix' - '训练文档矩阵'
    'trainCategory' - '训练标签向量'
@Returns : 
    'p0Vec' - '非侮辱类的条件概率数组'
    'p1Vec' - '侮辱类的条件概率数组'
    'pAbusive' - '文档输入侮辱类的概率'
@Time : ''2020/02/06
'''
def trainNB(trainMatrix, trainCategory):
    #训练集的样本数量
    numTrainDocs = len(trainMatrix)
    #词汇表的长度
    numWords = len(trainMatrix[0])
    #侮辱类文档的概率。先验概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    #创建长度位词汇表长度的0数组
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    #分母初始化为0，拉普拉斯平滑
    p0Denmo = 2.0
    p1Denmo = 2.0

    #遍历每个文档，统计数量
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denmo += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denmo += sum(trainMatrix[i])

    #取对数，防止溢出
    p0Vec = np.log(p0Num / p0Denmo)
    p1Vec = np.log(p1Num / p1Denmo)

    return p0Vec, p1Vec, pAbusive

'''
@Desc : '分类函数'
@Parameters :
    'vec2Classify' - '待分类的词组条'
    'p0Vec' - '非侮辱类词组的条件概率'
    'p1Vec' - '侮辱类词组的条件概率'
    'pClass' - '文档属于侮辱类的概率，即先验概率，pAb'
@Returns :
    '0' - '属于非侮辱类'
    '1' - '属于侮辱类'
@Time : '2020/02/06'
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    #对应元素相乘，log(A*B) = log A + log B，p1Vec已在前一个函数取过对数
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass)
    if p1 > p0:
        return 1
    else:
        return 0

'''
@Desc : '测试分类器'
@Parameter : 'none'
@Return : 'none'
@Time : '2020/02/06'
'''
def testNB():
    #创建实验样本
    postingList, classVec = loadDataSet()
    #创建词汇表
    vocabList = createVocaList(postingList)
    #训练分类器
    trainMat = []
    for wordSet in postingList:
        trainMat.append(word2Vec(wordSet, vocabList))

    p0V, p1V, pAb = trainNB(trainMat, classVec)

    #测试样本
    testEntry = ['love', 'my', 'dalmation']
    #测试样本向量化
    thisDoc = np.array(word2Vec(testEntry, vocabList))

    if classifyNB(thisDoc, p0V, p1V, pAb):
        print('侮辱类')
    else:
        print('非侮辱类')

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(word2Vec(testEntry, vocabList))                #测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print('属于侮辱类')                                        #执行分类并打印分类结果
    else:
        print('属于非侮辱类')

if __name__ == '__main__':
    testNB()