# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'speechFiltration.py'
@Time: '2020/02/06'
@Desc : 'Naive Bayes for speech filtering'
'''

import os
import jieba
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

'''
@Desc : '中文文本切分'
@Parameter :
    'folderPath' - '文件路径'
    'testSize' - '测试集占比，默认占全部的20%'
@Returns :
    'allWordsList' - '按词频降序排序的训练集列表'
    'trainDataList' - '训练集列表'
    'testDataList' - '测试集列表'
    'trainClassList' - '训练集标签'
    'testClassList' - '测试集标签'
@Time : '2020/02/07' 
'''
def textProcessing(folderPath, testSize=0.2):
    #将folderPath下的文件夹放入列表
    folderList = os.listdir(folderPath)
    #训练集
    dataList = []
    classList = []

    #遍历每个子文件夹
    for folder in folderList:
        #根据子文件夹，生成新的路径
        newFolderPath = os.path.join(folderPath, folder)
        #存放于子文件夹下的文件列表
        files = os.listdir(newFolderPath)

        j = 1
        #遍历每个txt文件
        for file in files:
            #每类txt样本最多100个
            if j > 100:
                break
            #打开txt文件
            with open(os.path.join(newFolderPath, file), 'r', encoding='utf-8') as f:
                raw = f.read()

            #分词，返回一个可迭代的generator
            wordCut = jieba.cut(raw, cut_all=False)
            wordList = list(wordCut)

            dataList.append(wordList)
            classList.append(folder)

            j += 1

    #zip压缩合并，将数据与标签对应压缩
    dataClassList = list(zip(dataList, classList))
    #数据集乱序
    random.shuffle(dataClassList)
    #训练集和测试集的切分处的索引值
    index = int(len(dataClassList) * testSize) + 1
    #训练集
    trainList = dataClassList[index:]
    #测试集
    testList = dataClassList[:index]
    #训练集和测试集压缩
    trainDataList, trainClassList = zip(*trainList)
    testDataList, testClassList = zip(*testList)

    #统计训练集词频
    allWordsdict = {}
    for wordList in trainDataList:
        for word in wordList:
            if word in allWordsdict.keys():
                allWordsdict[word] += 1
            else:
                allWordsdict[word] = 1

    #根据键的值倒序排序
    allWordsTupleList = sorted(allWordsdict.items(), key = lambda f:f[1], reverse=True)
    #解压缩
    allWordsList, allWordsNum = zip(*allWordsTupleList)
    allWordsList = list(allWordsList)

    return allWordsList, trainDataList, testDataList, trainClassList, testClassList

'''
@Desc : '读取文件内容，并去重'
@Parameter : 
    'path' - '文件路径'
@Return ：
    'wordsSet' - '读取到的内容的set集合'
@Time : '2020/02/07'
'''
def makeWordsSet(path):
    wordsSet = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            #去除回车
            word = line.strip()
            if len(word) > 0:
                wordsSet.add(word)

    return wordsSet

'''
@Desc : '文本特征选取'
@Parameter :
    'allWordsList' - '训练集所有文本列表'
    'deleteN' - '删除词频最高的N个词'
    'stopWordsSet' - '指定的结束语'
@Return :
    'featureWords' - '特征集'
'''
def wordsDict(allWordsList, deleteN, stopWordsSet = set()):
    featureWords = []
    n = 1

    for t in range(deleteN, len(allWordsList)):
        #特征集维度不超过1000
        if n > 1000:
            break
        #如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not allWordsList[t].isdigit() and allWordsList[t] not in stopWordsSet and 1 < len(allWordsList[t]) < 5 :
            featureWords.append(allWordsList[t])
        n +=1

    return featureWords

'''
@Desc : '将文本向量化'
@Parameters :
    'trainDataList' - '训练集'
    'testDataList' - '测试集'
    'featureWords' - '特征集'
@Returns :
    'trainFeatureList' - '训练集向量化的列表'
    'testFeatureList' - '测试集向量化的列表'
@Time : '2020/02/07'
'''
def textFeature(trainDataList, testDataList, featureWords):
    def text_feature(text, featureWords):
        textWords = set(text)
        features = [1 if word in textWords else 0 for word in featureWords]
        return features
    trainFeatureList = [text_feature(text, featureWords) for text in trainDataList]
    testFeatureList = [text_feature(text, featureWords) for text in testDataList]

    return trainFeatureList, testFeatureList

'''
@Desc : '新闻分类器'
@Parameters :
    'trainFeatureList' - '训练集向量化的特征文本'
    'testFeatureList' - '测试集向量化的特征文本'
    'trainClassList' - '训练集分类标签'
    'testClassList' - '测试集分类标签'
@Return :
    'testAccuracy' - '分类器精度'
@Time : '2020/02/07'
'''
def textClassifier(trainFeatureList, testFeatureList, trainClassList, testClassList):
    classifier = MultinomialNB().fit(trainFeatureList, trainClassList)
    testAccuracy = classifier.score(testFeatureList, testClassList)

    return testAccuracy

if __name__ == '__main__':
    #文本处理
    folderPath = './Sample'
    allWordsList, trainDataList, testDataList, trainClassList, testClassList = textProcessing(folderPath, testSize=0.2)

    #生成结束语列表
    stopWordsFile = './stopwords_cn.txt'
    stopWordsSet = makeWordsSet(stopWordsFile)

    #选取文本特征
    #featureWords = wordsDict(allWordsList, 100, stopWordsSet)

    testAccuracyList = []
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        featureWords = wordsDict(allWordsList, deleteN, stopWordsSet)
        trainFeatureList, testFeatureList = textFeature(trainDataList, testDataList, featureWords)
        testAccuracy = textClassifier(trainFeatureList, testFeatureList, trainClassList, testClassList)
        testAccuracyList.append(testAccuracy)

    plt.figure()
    plt.plot(deleteNs, testAccuracyList)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()