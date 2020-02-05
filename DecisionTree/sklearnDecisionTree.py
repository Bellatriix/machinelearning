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
import pydotplus
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO

if __name__ == '__main__':
    filename = r'./lenses.txt'
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate', 'class']
    lensesDataFrame = pd.read_table(filename, header=None, sep='	')
    #将标签设置为对应的列索引名
    lensesDataFrame.columns = lensesLabels

    lastColumn = lensesDataFrame.columns[len(lensesDataFrame.columns)-1]
    classLabels = lensesDataFrame[lastColumn].to_frame(name=lastColumn)
    dataSet = lensesDataFrame.drop(lastColumn, axis=1)

    #创建序列化对象
    le = LabelEncoder()
    #将数据序列化
    for col in dataSet.columns:
        dataSet[col] = le.fit_transform(dataSet[col])

    #构建决策树
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(dataSet.values.tolist(), classLabels)
    #dot_data = StringIO()
    #tree.export_graphviz(clf, out_file=dot_data, feature_names=dataSet.keys(),
    #                     class_names=clf.classes_, filled=True, rounded=True,
    #                     special_characters=True)
    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #graph.write_pdf("./tree.pdf")

    #使用决策树预测
    print(clf.predict([[1,1,1,0]]))