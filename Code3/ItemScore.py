import pandas as pd
import numpy as np
import time

# 记时
time_start = time.time()


'''导入数据'''
train = pd.read_csv('Traindata.csv',index_col=0)
test = pd.read_csv('Testdata.csv',index_col=0)
Sim_Score = pd.read_csv('相似顾客相似度表.csv', index_col=0)
print(Sim_Score)

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
CosScore = pd.DataFrame(index=Sim_Score.index,columns=train.columns).fillna(0)
for i in list(CosScore.index):
    s1 = Sim_Score.loc[i, list(Sim_Score.columns)[0:5]] # 用来保存最相似客户的ID
    s2 = Sim_Score.loc[i, list(Sim_Score.columns)[5:]] # 用来保存最相似客户对应的相似度得分
    ScoreList = np.array(list(0 for i in range(len(train.columns))), dtype='float64') # 用来保存相似度得分列表
    Total_s2 = 0 # 计算余弦相似度加权使用

    for j in range(len(s1)):
        if s1[j] != 0 and s2[j] != 0:
            c = np.array(train.loc[s1[j]], dtype='float64')*s2[j] # 以余弦相似度作为权重 计算指数评分
        ScoreList += c
        Total_s2 += np.array(s2[j])
    CosScore.loc[i] = ScoreList/Total_s2

print(CosScore)

CosScore.to_csv('Score.csv')





# 计时
time_end = time.time()
print('time cost', time_end - time_start, 's')