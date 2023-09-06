import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import xlwt
import time
import csv
import pandas as pd
import numpy as np
import time


# 记时
time_start = time.time()

'''读取训练集数据与测试集数据'''
train = pd.read_csv('Traindata.csv',index_col=0)
test = pd.read_csv('Testdata.csv',index_col=0)



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




'''将数据集进行标准化（逐列）'''
for i in list(train.columns):
    Mean = np.mean(train.loc[:, i])
    Std = np.std(train.loc[:, i])
    if Std != 0:
        train.loc[:, i] = (train.loc[:, i]-Mean)/Std
    else:
        continue
print(train)



'''使用推荐算法'''
# 算法思路，将最接近的K个客户所购买的商品数量标准分数，以余弦相似度为权重，计算加权平均数作为指数得分，最终得分最高的n类商品为推荐商品。
# 使用推荐评分算法
CosScore = pd.DataFrame(index=SimilarCos_Frame.index,columns=train.columns).fillna(0)
for i in list(CosScore.index):
    s1 = SimilarCos_Frame.loc[i, list(SimilarCos_Frame.columns)[0:5]] # 用来保存最相似客户的ID
    s2 = SimilarCos_Frame.loc[i, list(SimilarCos_Frame.columns)[5:]] # 用来保存最相似客户对应的相似度得分
    ScoreList = np.array(list(0 for i in range(len(train.columns))), dtype='float64') # 用来保存相似度得分列表
    Total_s2 = 0 # 计算余弦相似度加权使用

    for j in range(len(s1)):

        if s1[j] != 0 and s2[j] != 0:
            c = np.array(train.loc[s1[j]], dtype='float64')*s2[j] # 以余弦相似度作为权重 计算指数评分
        ScoreList += c
        Total_s2 += np.array(s2[j])
    CosScore.loc[i] = ScoreList/Total_s2



# 进行推荐商品
recommend_frame = pd.DataFrame(index=CosScore.index,columns=['Stock{}'.format(i for i in range(1,11))]).fillna(0)
for i in list(CosScore.index):
    Stock_score = CosScore.loc[i]
    Stock_score = Stock_score.sort_values(ascending=False)
    for k in range(1, 11):
        recommend_frame.loc[i, 'Stock{}'.format(k)] = Stock_score.index[k-1]

print(recommend_frame)

recommend_frame.to_csv('推荐商品_Test.csv')


'''评估算法准确性'''
Customer_buy = 0
Recommand_True = 0
for i in list(recommend_frame.index):
    Train_set = set(recommend_frame.loc[i])
    Test_set = set()
    for k in list(test.columns):
        if test.loc[i, k] != 0:
            Test_set.add(k)
    Customer_buy += len(list(Test_set))

    Result = list(Test_set.intersection(Train_set))

    Recommand_True += len(Result)

Recommand_All = len(recommend_frame.index) * (j - 1)
Recommand_Precison = Recommand_True / Recommand_All
print('Precison:{}'.format(Recommand_Precison))

# 计算推荐系统的召回率(使用推荐为正确的商品个数/顾客购买商品总数)
Recommand_Recall = Recommand_True / Customer_buy
print('Recall:{}'.format(Recommand_Recall))

# 计算F值
if Recommand_Recall!=0 and Recommand_Precison !=0:
    F_score = 2* Recommand_Precison * Recommand_Recall / (Recommand_Precison +Recommand_Recall)
else:
    F_score = 0
print('F:{}'.format(F_score))




# 计时
time_end = time.time()
print('time cost', time_end - time_start, 's')








