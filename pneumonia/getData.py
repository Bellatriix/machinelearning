# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : 'Zhang peng'
@Software : 'IDEA'
@File : 'getData.py'
@Time: '2020/2/1'
@Desc : 'Daily data of new coronavirus obtained by reptiles'
'''

from selenium import webdriver
import pandas as pd
from datetime import datetime,date,timedelta

'''
@Desc : 'Capture and analyze data'
@Parameters : 
    'output' - 'Storage file output path, without file name, the default name is date.xlsx'
@Returns : 
    'suc' - 'Daily data file output succeeded'
@Time : '2020/2/1'
'''
def getData(output):
    url = 'https://ncov.dxy.cn/ncovh5/view/pneumonia'

    driver = webdriver.Chrome()
    driver.get(url)

    div = driver.find_elements_by_css_selector('p[class*="subBlock1___j0DGa"]')
    webdriver.ActionChains(driver).move_to_element(div[8]).click(div[2]).perform()

    #获取数据集
    dataSetList = driver.find_elements_by_class_name('fold___xVOZX')

    yesterday_time = date.today() + timedelta(days=-1)
    now_time = yesterday_time.strftime('%Y.%m.%d')
    result = pd.DataFrame(columns=('time','province','city','diagnosis','death'))

    for dataSet in dataSetList:
        province = dataSet.find_element_by_class_name('subBlock1___j0DGa').text

        citySetList = dataSet.find_elements_by_class_name('areaBlock2___27vn7')

        #此循环不含香港台湾澳门西藏
        for cityLabel in citySetList:
            #if cityLabel:
            cities = cityLabel.find_elements_by_tag_name('p')

            if cities:
                city = cities[0].find_element_by_tag_name('span').get_attribute('innerHTML')
                diagnosis = cities[1].get_attribute('innerHTML')
                death = cities[2].get_attribute('innerHTML')

                newDataFrame = pd.DataFrame({
                    'time' : now_time,
                    'province' : province,
                    'city' : city,
                    'diagnosis' : diagnosis,
                    'death' : death
                }, index=[0])

                result = result.append(newDataFrame,ignore_index=True)

        #单独处理香港台湾澳门西藏
        if not(len(citySetList)):
            cities = dataSet.find_elements_by_class_name('areaBlock1___3V3UU')[0].find_elements_by_tag_name('p')
            city = cities[0].text
            diagnosis = cities[1].get_attribute('innerHTML')
            death = cities[2].get_attribute('innerHTML')

            newDataFrame = pd.DataFrame({
                'time' : now_time,
                'province' : province,
                'city' : city,
                'diagnosis' : diagnosis,
                'death' : death
            }, index=[0])

            result = result.append(newDataFrame,ignore_index=True)

    result.to_excel(output+now_time+'.xlsx', index=False)
    driver.quit()
    return 'suc'

if __name__ == '__main__':
    output = 'C:/Users/张鹏/Desktop/肺炎数据/'
    result = getData(output=output)
    print(result)
