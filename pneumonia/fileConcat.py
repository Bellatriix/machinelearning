# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'fileConcat.py'
@Time: '2020/02/07'
@Desc : 'File content left connection'
'''

import pandas as pd

'''
@Desc : '读取文件并左连接'
@Parameters ： 
    'path1' - '左连接的文件'
    'path2' - '被连接的文件'
    'output' - '输出路径'
@Return :
    'suc' - '操作成功'
@Time : '2020/02/07'
'''
def  fileLeftCon(path1, path2, output):
    file1 = pd.read_excel(path1)
    file2 = pd.read_excel(path2)

    result = pd.merge(file1, file2, how='left', sort=False, left_on='cityL', right_on='cityR')

    #计算确诊新增
    diagDiff = result['diagnosisL'] - result['diagnosisR']
    result.insert(result.shape[1], 'diagDiff', diagDiff)
    #计算死亡新增
    deathDiff = result['deathL'] - result['deathR']
    result.insert(result.shape[1], 'deathDiff', deathDiff)

    result.to_excel(output, index=False)

    return 'suc'

if __name__ == '__main__':
    path1 = r'C:/Users/张鹏/Desktop/肺炎数据/0207.xlsx'
    path2 = r'C:/Users/张鹏/Desktop/肺炎数据/0206.xlsx'
    output = r'C:/Users/张鹏/Desktop/肺炎数据/result07.xlsx'

    print(fileLeftCon(path1, path2, output))