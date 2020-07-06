# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'test.py'
@Time: '2020/3/18 18:15'
@Desc : ''
'''

import numpy as np
import pydotplus
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO

def predictTree(clf):
    test_set_orig = pd.read_excel(r'F:/gis/result/trainSetTest.xlsx')
    test_class_label = test_set_orig['患有胃肠疾病']

    test_set = test_set_orig.drop(['患有胃肠疾病'], axis=1)

    rows = test_set.shape[0]

    countplus = 0
    countmins = 0
    countone = 0
    for row in range(rows):
        test_data = test_set.loc[int(row), :].values.reshape(1, 28)

        pred_value = clf.predict(test_data)

        if pred_value == test_class_label[row]:
            countplus += 1
            if pred_value == 1:
                countone += 1
        else:
            countmins -= 1

    print(countplus, countmins, countone)

if __name__ == '__main__':

    train_set_orig = pd.read_excel(r'F:/gis/result/trainSetTest.xlsx')

    train_class_label = train_set_orig['患有胃肠疾病']

    train_set_orig = train_set_orig.drop(['患有胃肠疾病'], axis=1)

    le = LabelEncoder()
    #将数据序列化
    for col in train_set_orig.columns:
        train_set_orig[col] = le.fit_transform(train_set_orig[col])

    #构建决策树
    clf = tree.DecisionTreeClassifier(max_depth=28)
    clf = clf.fit(train_set_orig.values.tolist(), train_class_label)

    predictTree(clf)
