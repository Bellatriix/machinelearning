# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'emailSort.py'
@Time: '2020/02/07'
@Desc : 'Naive Bayes for email filtering'
'''

import re
import numpy as np
import random
from NaiveBayes import speechFiltration as sf

'''
@Desc : '将字符串转变为单词列表'
@Parameter : 
    'bigString' - '长字符串'
@Return :
    'result' - '单词列表'
@Time : '2020/02/07'
'''
def textParse(bigString):
    #将特殊符号作为切分单词标志，即非字母，非数字
    listOfTokens = re.split(r'\W+', bigString)
    #除了单个字母，其他字母变小写
    result = [tok.lower() for tok in listOfTokens if len(tok) > 2]

    return  result

if __name__ == '__main__':
    docList = []
    classList = []

    for i in range(1, 26):
        #遍历读取文件，并将字符串切分为列表
        wordList = textParse(open('./spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        classList.append(1)

        wordList = textParse(open('./ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        classList.append(0)

    vocabList = sf.createVocaList(docList)

    #创建列表存储训练集索引和测试集索引
    trainSet = list(range(50))
    testSet = []

    #从50个数据中，随机选取40个做训练集，10个做测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])

    #创建训练集矩阵和分类标签
    trainMat = []
    trainClass = []

    for docIndex in trainSet:
        trainMat.append(sf.word2Vec(docList[docIndex], vocabList))
        trainClass.append(classList[docIndex])

    #训练算法
    p0V, p1V, pSpam = sf.trainNB(np.array(trainMat), np.array(trainClass))

    #错误分类计数
    errorCount = 0
    for docIndex in testSet:
        wordVector = sf.word2Vec(docList[docIndex], vocabList)
        if sf.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('分类错误的测试集：', docList[docIndex])

    print('错误率：%.2f%%' % float(float(errorCount) / len(testSet) * 100))