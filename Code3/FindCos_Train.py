import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import xlwt
import time
import csv


# 记时
time_start = time.time()

'''读取训练集数据与测试集数据'''
train = pd.read_csv('Traindata.csv',index_col=0)
test = pd.read_csv('Testdata.csv',index_col=0)




# 将数据逐行标准化
# for i in list(data.index):
#     Mean = np.mean(data.loc[i])
#     Std = np.std(data.loc[i])
#     data.loc[i] = (data.loc[i]-Mean)/Std
# print(data)

# '''标准化过程'''
# # 将训练集数据逐列标准化
# for i in list(train.columns):
#     Mean = np.mean(train.loc[:, i])
#     Std = np.std(train.loc[:, i])
#     if Std != 0:
#         train.loc[:, i] = (train.loc[:, i]-Mean)/Std
#     else:
#         continue
# print(train)

'''寻找测试集与训练集的交集'''
train_set = set()
test_set = set()

for i in list(train.index):
    train_set.add(i)

for i in list(test.index):
    test_set.add(i)

test_train = list(train_set.intersection(test_set))
print(test_train)


'''定义距离函数'''
# 定义欧氏距离
def euSimilar(a,b):
    dist = np.linalg.norm(a - b)
    return 1 / (dist+0.0001)

# 定义余弦相似度
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos


'''寻找K个相似客户'''
# 寻找最相似的K个客户
SimilarCos = dict() # 创建一个字典存储最相似的客户与得分
for i in test_train:
    distance_list = [0 for h in range(5)]
    SimilarCos_list = [0 for k in range(5)]
    for j in list(train.index.drop(i)):
        x = np.array(train.loc[i])
        y = np.array(train.loc[j])
        distance = cos_sim(x, y) # 计算余弦相似度
        if distance > min(distance_list):
            a = distance_list.index(min(distance_list))
            distance_list[a] = distance
            SimilarCos_list[a] = j
        else:
            continue

    Total_Simsocre = np.array(distance_list, dtype='float64').sum()  # 对相似度得分进行求和
    SimScore = [k/Total_Simsocre for k in distance_list] # 求得平均相似度得分

    SimilarCos[i] = [SimilarCos_list, SimScore]

print(SimilarCos)

# 将最相似顾客与其得分写入Excel表格中
rowname=list()
for headers in SimilarCos.keys():
    rowname.append(headers)
SimilarCos_Frame = pd.DataFrame(index=rowname, columns=['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5',
                                                        'Score1', 'Score2','Score3', 'Score4', 'Score5' ]).fillna(0)
for keys in SimilarCos.keys():
    list1 = SimilarCos[keys][0]+list(SimilarCos[keys][1])
    SimilarCos_Frame.loc[keys] = list1

print(SimilarCos_Frame)
SimilarCos_Frame.to_csv('相似顾客相似度表_Train.csv')



# 计时
time_end = time.time()
print('time cost', time_end - time_start, 's')










