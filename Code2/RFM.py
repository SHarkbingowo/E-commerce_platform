#导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import squarify

full_data = pd.read_csv('D:/data mining/第五届MAS案例大赛/附件3/records.csv')
#数据整体信息预览和无效值统计（没有无效值）
print("Dataset Information :", full_data.info(), end= '\n\n')
print("Sum of null values :", full_data.isna().sum())

# 负销量直方图占比（退单）
Negative_Quantity= full_data[full_data['Quantity']<0].Quantity
Sum_Total_Quantity= full_data['Quantity'].sum()
Percentage_Incorrect_Quantity= round((abs(Negative_Quantity.sum())/Sum_Total_Quantity), 2)*100
print("Percentage of Incorrect Quantity values: ",Percentage_Incorrect_Quantity, "%", end='\n\n')
plt.hist(Negative_Quantity, range= (-40, 0))
plt.title('Negative Quantity')
plt.show()

# 缺失值处理
full_data.dropna(axis=0, how='any', inplace=True)
# # 退单数据处理
# full_data = full_data[~full_data["InvoiceNo"].str.contains("C", na=False)]
print(full_data.info())

# 交易日期  记录购买日期.max()+1day（什么意思？）
snapshot_date = pd.to_datetime(full_data['InvoiceDate']).max() + timedelta(days=1)
print(snapshot_date)
# 客户消费总额
full_data["CostPerOrder"] = full_data["Quantity"] * full_data["UnitPrice"]
# Changing datatype of InvoiceDate type
full_data["InvoiceDate"] = pd.to_datetime(full_data["InvoiceDate"])
print(full_data.info())

dg = (full_data
    .groupby(by='CustomerID', as_index=False)
    .agg(
    {'InvoiceDate': [lambda date: (snapshot_date - date.min()).days, lambda date: (snapshot_date - date.max()).days]
        , 'InvoiceNo': lambda num: num.count()
        , 'Quantity': lambda quant: quant.sum()
        , 'CostPerOrder': lambda price: price.sum()
     })
)
# 改写表格数据栏变量名 num_days距离上次购物时间、most_recent_txn最近一次购物时间、num_orders订单数、num_units购买物品数、total_revenue总购买额
dg.columns= ['CustomerID', 'num_days', 'most_recent_txn', 'num_orders', 'num_units', 'total_revenue']
dg.head()

dg = dg[dg['total_revenue']> 0]

print(dg.info())

# Customer Life Time Value顾客终身价值https://zhuanlan.zhihu.com/p/135195797

#平均每单金额
dg['avg_order_value']= dg['total_revenue']/dg['num_orders']
print(dg.head())
#购买频率
purchase_frequency=sum(dg['num_orders'])/dg.shape[0]
#重复率
repeat_rate= dg[dg.num_orders > 1].shape[0]/dg.shape[0]
#流失率=1-重复率
churn_rate= 1 - repeat_rate
purchase_frequency,repeat_rate,churn_rate

#边际收益（成本位置啊？？？）
dg['profit_margin']= dg['total_revenue']*0.25
# 顾客平均价值
dg['CLTV']= (dg['avg_order_value']*purchase_frequency)/churn_rate
#顾客终身价值
dg['cust_lifetime_value']= dg['CLTV']*dg['profit_margin']
dg.head()

# most_recent_txn -> recency; num_orders -> frequency; total_revenue -> monetary value
# RFM分布图展示
plt.figure(figsize=(12,10))
# R分布
plt.subplot(3, 1, 1); sns.distplot(dg['most_recent_txn'])
# F分布
plt.subplot(3, 1, 2); sns.distplot(dg['num_orders'])
# M分布
plt.subplot(3, 1, 3); sns.distplot(dg['total_revenue'])
plt.show()

# --RFM分组--
#创建标签rfm和范围划定
r_labels = range(1, 5)         # Recency
f_labels = range(4, 0, -1)     # Frequency
m_labels = range(4, 0, -1)     # Monetary Value
r_groups = pd.qcut(dg['most_recent_txn'], q= 4, labels = r_labels)
# 将这些标签分配给4个相等的百分比组
f_groups = pd.qcut(dg['num_orders'], q= 4, labels = f_labels)
# 将这些标签分配给4个相等的百分比组
m_groups = pd.qcut(dg['total_revenue'], q= 4, labels = m_labels)
# 将新创建的列分配给数据帧
dg = dg.assign(R= r_groups.values, F= f_groups.values, M= m_groups.values)
dg.rename({'most_recent_txn': 'recency', 'num_orders': 'frequency', 'total_revenue': 'monetary'}, axis=1, inplace=True)
print(dg.dtypes)

print(dg.info())
#分组后的RFM展示
plt.figure(figsize=(12,10))
# R
plt.subplot(3, 1, 1); sns.distplot(dg['R'])
# F
plt.subplot(3, 1, 2); sns.distplot(dg['F'])
# M
plt.subplot(3, 1, 3); sns.distplot(dg['M'])
plt.show()

# RFM分类
def join_rfm(df):
    strs = [str(df['R']), str(df['F']), str(df['M']) ]
    return ''.join(strs)
dg['RFM_segment'] = dg.apply(join_rfm, axis= 1)
print(dg.head())

# 客户分类统计和条形图展示
rfm_count_unique = dg['RFM_segment'].nunique()
print(rfm_count_unique)
rfm_seg_order = dg['RFM_segment'].value_counts().sort_index().index
plt.figure(figsize=(20,5))
sns.set_palette("Set2")
ax= sns.countplot(data= dg, x='RFM_segment', dodge= False, order= rfm_seg_order)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");
plt.show()

# 对几种客户添加标签 Core核心客户 Loyal忠诚用户 Whales高额用户 promisiong发展用户
def user_segment(dg):
    X = [1, 2, 3, 4]
    if ((dg['F'] == 1) & (dg['R'] == 1) & (dg['M'] == 1)) &(dg['RFM_segment'] == '1.01.01.0'):
        return 'Core'
    elif ((dg['F'] == 1) & (dg['R'] <= 2) & (dg['M'] == 1)) & (dg['RFM_segment'] != '1.01.01.0'):
        return 'Loyal'
    elif ((dg['M'] == 1) & (dg['R'] >= 1) & (dg['F'] <= 2)) & (dg['RFM_segment'] != '1.01.01.0'):
        return 'Whales'
    # elif (dg['F'] == 1) & (dg['M'] == 3 | dg['M'] == 4) & (dg['R'] >= 1):
    #     return 'Promising'
    elif (dg['F'] == 1) & (dg['M'] >= 2) & (dg['R'] >= 1):
        return 'Promising'
    elif (dg['F'] == 2) & (dg['M'] >= 1) & (dg['R'] >= 1):
        return 'Normal'
    elif (dg['R'] == 1) & (dg['F'] == 4) & (dg['M'] >= 1):
        return 'Rookies'
    elif (dg['R'] == 4) & (dg['F'] == 4) & (dg['M'] >= 1):
        return 'Slipping'
    else:
        return 'Normal'

dg['user_segment'] = dg.apply(user_segment, axis=1)
dg.head()

# 计算各类客户的数据信息
user_segment_size = (
                    dg.
                        groupby('user_segment')
                        .agg({
                                'CustomerID': 'count',
                                'recency': 'mean',
                                'frequency': 'mean',
                                'monetary': ['all', 'count']
                        }).round(1)
                )
#展示结果
user_segment_size.head()

user_segment_size.to_csv('D:/data mining/第五届MAS案例大赛/附件3/客户分类结果1.csv',encoding = 'utf_8_sig')

# 用占比的成分图来表示
user_segment_size.columns = ['Customers','RecencyMean','FrequencyMean','MonetaryMean', 'RevenueCount']
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes= user_segment_size['RevenueCount'],
              label= user_segment_size.index, alpha=.7 )
plt.title("proportion of customers num. at different levels",fontsize=18,fontweight="bold")
plt.axis('on')
plt.show()

fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes= user_segment_size['MonetaryMean'],
              label= user_segment_size.index, alpha=.7 )
plt.title("proportion of customers sales at different levels",fontsize=18,fontweight="bold")
plt.axis('on')
plt.show()

# 不用层级用户人数条形图
rfm_score_order = dg['user_segment'].value_counts().index
plt.figure(figsize=(20,5))
sns.set_palette("Set2")
ax= sns.countplot(data= dg, x='user_segment', dodge= False, order= rfm_score_order)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
plt.show()

# result_df = dg.merge(full_data, on='CustomerID', how='left')
#
# def add_date_dim(df, col_name):
#     df['year'] = pd.DatetimeIndex(df[col_name]).year
#     df['month'] = pd.DatetimeIndex(df[col_name]).month
#     df['quarter'] = pd.DatetimeIndex(df[col_name]).quarter
#     df['week'] = pd.DatetimeIndex(df[col_name]).week
#     df['weekday'] = pd.DatetimeIndex(df[col_name]).weekday
#     return df
#
# add_date_dim(result_df, 'InvoiceDate')
# result_df.to_csv('D:/data mining/第五届MAS案例大赛/附件3/result.csv',encoding = 'utf_8_sig')

