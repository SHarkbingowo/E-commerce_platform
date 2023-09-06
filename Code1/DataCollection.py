import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import xlwt
import time

# 记时
time_start = time.time()

# 解决绘图区中文乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''导入时间数据'''
data = pd.read_excel('../ecommercedata-历史订单.xlsx',
                     usecols=['InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate',
                              'UnitPrice', 'InvoiceDate', 'CustomerID'])
print('------------查看数据大小及原始数据信息-----------------')

data.info()
data.describe()

'''查看数据缺失值'''
total_null = data.isnull().sum().sort_values(ascending=False)
print('-------------查看数据缺失值情况-------------------------')
print(total_null)  # 可以看到该数据不存在缺失值

'''处理时间数据'''
Total_Time = ((pd.to_datetime('2011/11/30 17:42:00 ', format='%Y%m%d %H:%M:%S') -
               pd.to_datetime('2010/12/1 8:26:00', format='%Y%m%d %H:%M:%S')).seconds) / 60  # 设置总时间周期
for date in data['InvoiceDate']:
    date = pd.to_datetime(date, format='%Y%m%d %H:%M:%S')

# # 时间间隔
# print((data['InvoiceDate'][1]-data['InvoiceDate'][0]).seconds)
#
# # 时间最大值
# print(max([data['InvoiceDate'][1],data['InvoiceDate'][0]]))


'''将商品进行汇总'''
# 查看商品的所有类型编码
costype = set()
for i in data['StockCode']:
    costype.add(i)
costype = list(costype)  # 将集合转变为列表

'''将退单与订单分开'''
Certain_ord = data.loc[~(data['InvoiceNo'].str.len() == 7)]
Cancel_ord = data.loc[~(data['InvoiceNo'].str.len() != 7)]
print('-----------------查看订单与退单情况--------------------')
print(Certain_ord)
print(Cancel_ord)

# 创建dataframe来汇总商品信息
cosinfoframe = pd.DataFrame(index=costype,
                            columns=['QuantitySum', 'SalesSum', 'ChargeBack',
                                     'ChargeRatio', 'TimeLong', 'SalesperMin',
                                     'QuantityperMin', 'Customer']).fillna(0)

# 设置计算销售总量 时间长度 销售额总量 购买客户数
for j in costype:
    Time = []
    Customer = set()
    for i in range(len(data)):
        if j == data['StockCode'][i]:
            # 计算销售总量
            cosinfoframe.loc[j, 'QuantitySum'] += data['Quantity'][i]

            # 添加时间
            Time.append(data['InvoiceDate'][i])

            # 计算销售额总量
            Sales = data['Quantity'][i] * data['UnitPrice'][i]
            cosinfoframe.loc[j, 'SalesSum'] += Sales

            # 添加客户ID
            Customer.add(data['CustomerID'][i])

        else:
            continue

        # 计算时间长度
        TimeLong = ((max(Time) - min(Time)).seconds) / 60
        if TimeLong != 0:
            cosinfoframe.loc[j, 'TimeLong'] = TimeLong
        elif TimeLong == 0:
            cosinfoframe.loc[j, 'TimeLong'] = Total_Time  # 若只进行过一次交易，则时间为总周期长度

        # 计算购买人数
        cosinfoframe.loc[j, 'Customer'] = len(Customer)

print('1 完成')

# 设置计算每单位分钟销售量以及每单位分钟销售额
for j in costype:
    cosinfoframe.loc[j, 'QuantityperMin'] = cosinfoframe.loc[j, 'QuantitySum'] / cosinfoframe.loc[j, 'TimeLong']
    cosinfoframe.loc[j, 'SalesperMin'] = cosinfoframe.loc[j, 'SalesSum'] / cosinfoframe.loc[j, 'TimeLong']
print('2 完成')

# 设置计算退单率
for j in costype:
    for i in Cancel_ord.index.tolist():
        if j == data['StockCode'][i]:
            cosinfoframe.loc[j, 'ChargeBack'] += abs(Cancel_ord['Quantity'][i])

for j in costype:
    cosinfoframe.loc[j, 'ChargeRatio'] = cosinfoframe.loc[j, 'ChargeBack'] / (cosinfoframe.loc[j, 'ChargeBack'] +
                                                                              cosinfoframe.loc[j, 'QuantitySum'])
print('3 完成')

# 去除TimeLong特征变量
cosinfoframe.drop(['TimeLong'], axis=1)

# 查看汇总信息表
print('----------------查看汇总表信息')
print(cosinfoframe)

# 将汇总信息表写入excel
# file_name = "D:\研究生课程\研究生案例大赛\实验\\cosinfo.xlsx"
# workbook = xlsxwriter.Workbook(file_name)
# worksheet = workbook.add_worksheet('first_sheet')
# worksheet.write(cosinfoframe)
# workbook.close()

cosinfoframe.to_excel('Code1/cosinfoframe.xls')

# 计时
time_end = time.time()

print('time cost', time_end - time_start, 's')

# time cost 6923.7291848659515 s
