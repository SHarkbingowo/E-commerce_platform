import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import xlwt
import time
from scipy import stats
from hist import draw_hist
from Scatter import draw_Scatter
from sklearn.neighbors import LocalOutlierFactor
from Scatter import draw_outlierScatter


# 记时
time_start = time.time()

# 解决绘图区中文乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"""导入数据"""
data = pd.read_csv('Code1/cosinfoframe.csv')
print(data)
print(data.shape)

"""数据预处理"""
# 由于有的订单数量为负数，说明有可能是因为数据周期不完全，在这个周期内只收录了退订单的信息，所以首先把订单数量为负数商品的进行删除。
data_precondict = data.loc[~(data['QuantitySum'] <= 0)]
data_drop = data.loc[~(data['QuantitySum'] > 0)]  # 将丢失的数据收集

# 由于有的订单销售额为负数，所以将销售额为负数的商品也不计入考虑范围内。
data_drop = data_drop.append(data_precondict.loc[~(data_precondict['SalesSum'] > 0)])
data_precondict = data_precondict.loc[~(data_precondict['SalesSum'] <= 0)]



'''绘制散点图来查看数据分布'''
# 绘制散点图1
plt.figure(figsize=(15, 12))
plt.subplot(2, 2, 1)
draw_Scatter(data_precondict['QuantitySum'], '总销量')

plt.subplot(2, 2, 2)
draw_Scatter(data_precondict['SalesSum'], '总销售额')

plt.subplot(2, 2, 3)
draw_Scatter(data_precondict['ChargeBack'], '退订单总量')

plt.subplot(2, 2, 4)
draw_Scatter(data_precondict['ChargeRatio'], '退单率')

# 调整子图的位置
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

plt.savefig('../pic/Scatter1')

# 绘制散点图2
plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
draw_Scatter(data_precondict['SalesperMin'], '每分钟销售额')

plt.subplot(2, 2, 2)
draw_Scatter(data_precondict['QuantityperMin'], '每分钟销售量')

plt.subplot(2, 2, 3)
draw_Scatter(data_precondict['Customer'], '购买顾客数')

# 调整子图的位置
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

plt.savefig('../pic/Scatter2')

# 通过散点图的绘制可以看到 有些特征数据存在一定的异常值，异常值的存在可能会对后面的聚类算法造成一定的影响，所以需要对异常值进行处理。

'''处理数据的异常值'''
# 设置lof算法
clf = LocalOutlierFactor(n_neighbors=25, contamination=0.005)
labels = clf.fit_predict(data_precondict.iloc[:, 1:])

data_drop = data_drop.append(data_precondict[labels == -1])
data_precondict = data_precondict[labels > 0]


# 绘制异常点散点图：
plt.figure(figsize=(15, 12))
plt.subplot(2, 2, 1)
draw_outlierScatter(data=data_precondict, data_drop=data_drop,variable='SalesSum',varnum='每分钟销售额')
plt.subplot(2, 2, 2)
draw_outlierScatter(data=data_precondict, data_drop=data_drop,variable='QuantityperMin',varnum='每分钟销售量')
plt.subplot(2, 2, 3)
draw_outlierScatter(data=data_precondict, data_drop=data_drop,variable='Customer',varnum='购买顾客数')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
plt.savefig('../pic/Scatter4')


plt.figure(figsize=(15, 12))
plt.subplot(2, 2, 1)
draw_outlierScatter(data=data_precondict, data_drop=data_drop,variable='SalesperMin',varnum='总销量')
plt.subplot(2, 2, 2)
draw_outlierScatter(data=data_precondict, data_drop=data_drop,variable='QuantitySum',varnum='总销售额')
plt.subplot(2, 2, 3)
draw_outlierScatter(data=data_precondict, data_drop=data_drop,variable='ChargeBack',varnum='退订单总量')
plt.subplot(2, 2, 4)
draw_outlierScatter(data=data_precondict, data_drop=data_drop,variable='ChargeRatio',varnum='退单率')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
plt.savefig('../pic/Scatter3')


data_precondict.to_excel('cosinfoframe_precondit.xls')  # 将预处理过的数据存入excel表格
data_drop.to_excel('../cosinfo/data_drop1.xls')


'''查看数据缺失值'''
total_null = data_precondict.isnull().sum().sort_values(ascending=False)
print('-------------查看数据缺失值情况-------------------------')
print(total_null)  # 可以看到该数据不存在缺失值
#
'''进行特征的概括性度量'''
# 创建表格记录各个特征的概括性度量
Stats = ['平均数', '下四分位数', '中位数', '上四分位数', '众数', '标准差', '偏态系数', '峰态系数']
StatsFrame = pd.DataFrame(index=Stats,
                          columns=['QuantitySum', 'SalesSum', 'ChargeBack',
                                   'ChargeRatio', 'SalesperMin',
                                   'QuantityperMin', 'Customer']).fillna(0)

for i in list(data_precondict.columns)[1:]:
    StatsFrame.loc['平均数', i] = np.mean(data_precondict[i])
    StatsFrame.loc['下四分位数', i] = np.quantile(data_precondict[i], 0.25)
    StatsFrame.loc['中位数', i] = np.median(data_precondict[i])
    StatsFrame.loc['上四分位数', i] = np.quantile(data_precondict[i], 0.75)
    StatsFrame.loc['众数', i] = stats.mode(data_precondict[i])[0][0]
    StatsFrame.loc['标准差', i] = np.std(data_precondict[i])
    StatsFrame.loc['偏态系数', i] = stats.skew(data_precondict[i])
    StatsFrame.loc['峰态系数', i] = stats.kurtosis(data_precondict[i])

StatsFrame.to_excel('../cosinfo/CosinfoStat.xlsx')

'''数据的标准化'''
for i in list(data_precondict.columns[1:]):
    Mean = np.mean(data_precondict[i])
    Stdv = np.std(data_precondict[i])
    for j in list(data_precondict[i].index):
        data_precondict.loc[j,i] = (data_precondict.loc[j,i]-Mean)/Stdv

print(data_precondict)

data_precondict.to_excel('../cosinfo/cosinfoframe_Zscore.xlsx')

# 创建表格记录各个特征的概括性度量
Stats = ['平均数', '下四分位数', '中位数', '上四分位数', '众数', '标准差', '偏态系数', '峰态系数']
StatsFrame = pd.DataFrame(index=Stats,
                          columns=['QuantitySum', 'SalesSum', 'ChargeBack',
                                   'ChargeRatio', 'SalesperMin',
                                   'QuantityperMin', 'Customer']).fillna(0)

for i in list(data_precondict.columns)[1:]:
    StatsFrame.loc['平均数', i] = np.mean(data_precondict[i])
    StatsFrame.loc['下四分位数', i] = np.quantile(data_precondict[i], 0.25)
    StatsFrame.loc['中位数', i] = np.median(data_precondict[i])
    StatsFrame.loc['上四分位数', i] = np.quantile(data_precondict[i], 0.75)
    StatsFrame.loc['众数', i] = stats.mode(data_precondict[i])[0][0]
    StatsFrame.loc['标准差', i] = np.std(data_precondict[i])
    StatsFrame.loc['偏态系数', i] = stats.skew(data_precondict[i])
    StatsFrame.loc['峰态系数', i] = stats.kurtosis(data_precondict[i])

StatsFrame.to_excel('../cosinfo/CosinfoStat1.xlsx')




'''绘制标准化后的直方图来查看特征分布'''
# 绘制直方图1
plt.figure(figsize=(15, 12))
plt.subplot(2, 2, 1)
draw_hist(data_precondict['QuantitySum'], '总销量')

plt.subplot(2, 2, 2)
draw_hist(data_precondict['SalesSum'], '总销售额')

plt.subplot(2, 2, 3)
draw_hist(data_precondict['ChargeBack'], '退订单总量')

plt.subplot(2, 2, 4)
draw_hist(data_precondict['ChargeRatio'], '退单率')

# 调整子图的位置
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

plt.savefig('../pic/hist1')

# 绘制直方图2
plt.figure(figsize=(15, 12))

plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
draw_hist(data_precondict['SalesperMin'], '每分钟销售额')

plt.subplot(2, 2, 2)
draw_hist(data_precondict['QuantityperMin'], '每分钟销售量')

plt.subplot(2, 2, 3)
draw_hist(data_precondict['Customer'], '购买顾客数')

# 调整子图的位置
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)

plt.savefig('../pic/hist2')

# 计时
time_end = time.time()
print('time cost', time_end - time_start, 's')
