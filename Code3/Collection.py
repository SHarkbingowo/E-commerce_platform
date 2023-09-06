import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import xlwt
import time

# 记时
time_start=time.time()


#解决绘图区中文乱码的问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

'''导入时间数据'''
data = pd.read_excel('..\ecommercedata-历史订单.xlsx',
                 usecols=['InvoiceNo','StockCode','Quantity','InvoiceDate',
                          'UnitPrice','InvoiceDate','CustomerID'])

'''划分训练集与验证集'''
data_train = data.iloc[:round(3*len(data.index)/4), :]
data_test = data.iloc[round(3*len(data.index)/4):, :]

print(data_train)
print(data_test)


'''处理训练集数据'''
'''汇总商品与顾客信息'''
item = set()
customer = set()
for i in list(data_train.index):
    item.add(data_train.loc[i, 'StockCode'])
    customer.add(data_train.loc[i,'CustomerID'])
item = list(item)
customer = list(customer)


'''制作顾客商品信息表'''
CosDataFrame_train = pd.DataFrame(index=customer, columns=item).fillna(0)

for i in list(data_train.index):
    itemId = data_train.loc[i,'StockCode']
    CustomerId = data_train.loc[i, 'CustomerID']
    Quantity = data_train.loc[i, 'Quantity']
    CosDataFrame_train.loc[CustomerId, itemId] += Quantity

print(CosDataFrame_train)

CosDataFrame_train.to_csv('Traindata.csv')


'''处理训练集数据'''
'''汇总商品与顾客信息'''
item = set()
customer = set()
for i in list(data_test.index):
    item.add(data_test.loc[i, 'StockCode'])
    customer.add(data_test.loc[i,'CustomerID'])
item = list(item)
customer = list(customer)


'''制作顾客商品信息表'''
CosDataFrame_test = pd.DataFrame(index=customer, columns=item).fillna(0)

for i in list(data_test.index):
    itemId = data_test.loc[i,'StockCode']
    CustomerId = data_test.loc[i, 'CustomerID']
    Quantity = data_test.loc[i, 'Quantity']
    CosDataFrame_test .loc[CustomerId, itemId] += Quantity

print(CosDataFrame_test)

CosDataFrame_test .to_csv('Testdata.csv')


# 计时
time_end=time.time()

print('time cost',time_end-time_start,'s')

